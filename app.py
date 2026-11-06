import argparse
from pathlib import Path

import cv2
import gradio as gr
import joblib
import numpy as np

from baldseg.core import BALD, GRADES, HAIR, VIEWS, bald_ratio, clahe, grader_features, predict, stage_label
from baldseg.face import FaceMasker, remove_face
from baldseg.norwood_art import illustration

COLORS = {HAIR: (0, 255, 0), BALD: (255, 33, 33)}


def overlay(rgb, cmap):
    color = np.zeros_like(rgb)
    for cls, c in COLORS.items():
        color[cmap == cls] = c
    return np.where(cmap[..., None] > 0, (0.55 * rgb + 0.45 * color).astype(np.uint8), rgb)


def heatmap(unc):
    return cv2.cvtColor(cv2.applyColorMap((unc * 255).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def plan_a(weights, conf, masker):
    from ultralytics import YOLO
    model = YOLO(weights / "best.pt")
    tree = joblib.load(weights / "grader.joblib")

    def run(rgb, view):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cmap = remove_face(predict(model, bgr, conf), masker(clahe(bgr), view))
        probs = np.zeros(len(GRADES))
        probs[tree.classes_] = tree.predict_proba([grader_features(cmap, view)])[0]
        return cmap, probs, None, f"{bald_ratio(cmap):.1%}"

    return run


def plan_b(ckpt, mc, masker):
    import torch
    from baldseg.multitask import load, predict as predict_b
    model, size = load(ckpt, "cuda" if torch.cuda.is_available() else "cpu")

    def run(rgb, view):
        bgr = clahe(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        out = predict_b(model, bgr, size, mc)
        cmap = remove_face(out["cmap"], masker(bgr, view))
        return cmap, out["grade_probs"], heatmap(out["uncertainty"]), f"{bald_ratio(cmap):.1%}"

    return run


def assess(plans, plan, images):
    shots = [(view, img) for view, img in zip(VIEWS, images) if img is not None]
    if not shots:
        raise gr.Error("Hãy tải lên ít nhất một ảnh.")
    results = [(view, img, *plans[plan](img, view)) for view, img in shots]
    probs = np.mean([r[3] for r in results], axis=0)
    level = int(probs.argmax())
    top = next((r for r in results if r[0] == "Top-Down"), None)
    label, vertex = stage_label(level, top[2], "Top-Down") if top else (GRADES[level], False)
    per_view = "\n".join(f"{view}: R = {ratio}, {GRADES[int(p.argmax())]} ({p.max():.0%})" for view, _, _, p, _, ratio in results)
    detail = " | ".join(f"{g}: {p:.0%}" for g, p in zip(GRADES, probs))
    seg = [(overlay(img, cmap), view) for view, img, cmap, _, _, _ in results]
    unc = [(u, view) for view, _, _, _, u, _ in results if u is not None]
    return seg, unc, f"Norwood {label}  (gộp {len(results)} góc chụp)\n{detail}", per_view, illustration(level + 2, vertex)


def examples(root, plan):
    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    rows = [[next((str(f) for f in d.glob(f"{v}.jpg")), None) for v in VIEWS] + [plan] for d in dirs]
    labels = [f"Nhãn Norwood {d.name[7]} (người {d.name.split('_')[1]})" if d.name.startswith("norwood") else f"Tập phân đoạn ({d.name})" for d in dirs]
    return rows, labels


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights-a", type=Path, default=Path("weights"))
    p.add_argument("--ckpt-b", type=Path, default=Path("weights_b/norwood.pt"))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--mc", type=int, default=8)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--examples", type=Path, default=Path("examples"))
    args = p.parse_args()
    import torch
    masker = FaceMasker(device="cuda" if torch.cuda.is_available() else "cpu")
    plans = {}
    if args.ckpt_b.exists():
        plans["B - SegFormer đa nhiệm"] = plan_b(args.ckpt_b, args.mc, masker)
    if (args.weights_a / "grader.joblib").exists():
        plans["A - YOLO11-seg"] = plan_a(args.weights_a, args.conf, masker)

    with gr.Blocks(title="Đánh giá rụng tóc Hamilton-Norwood") as demo:
        gr.Markdown("# Đánh giá mức độ rụng tóc theo thang Hamilton-Norwood\nTải lên 1–5 ảnh của cùng một người, mỗi ô một góc chụp. Ảnh đỉnh đầu (Top-Down) và phía trước (Front) cho kết quả tốt nhất; kết quả là trung bình xác suất của các góc.")
        with gr.Row():
            images = [gr.Image(type="numpy", label=v, height=180) for v in VIEWS]
        with gr.Row():
            plan = gr.Radio(list(plans), value=next(iter(plans)), label="Mô hình")
            button = gr.Button("Đánh giá", variant="primary")
        with gr.Row():
            with gr.Column(scale=2):
                stage = gr.Textbox(label="Mức độ (Hamilton-Norwood)", lines=2)
                per_view = gr.Textbox(label="Theo từng góc chụp", lines=5)
            art = gr.Image(label="Minh họa stage (nhìn từ đỉnh đầu)", height=280)
        seg = gr.Gallery(label="Phân đoạn (xanh: tóc, đỏ: da đầu lộ; mặt, tai, cổ đã loại khỏi phép tính)", columns=5, height=240)
        unc = gr.Gallery(label="Bản đồ bất định (Phương án B)", columns=5, height=240)
        rows, labels = examples(args.examples, plan.value)
        gr.Examples(rows, images + [plan], example_labels=labels, label="Ảnh mẫu từ tập test (mỗi dòng là một người)")
        button.click(lambda pl, *imgs: assess(plans, pl, imgs), [plan] + images, [seg, unc, stage, per_view, art])
    demo.launch(server_name="0.0.0.0", server_port=args.port)
