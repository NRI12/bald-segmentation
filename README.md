# Bald segmentation (ảnh đỉnh đầu)

Đánh giá mức độ rụng tóc kiểu nam theo thang Hamilton-Norwood từ **một ảnh chụp từ trên xuống đỉnh đầu**: phân đoạn tóc / da đầu lộ bằng SegFormer-B2 hoặc YOLO11m-seg (để so sánh), tính tỷ lệ da đầu lộ `R = S_bald / (S_bald + S_hair)`, so với ngưỡng R đo trên hình nhìn từ trên xuống của thang Hamilton-Norwood để ra mức.

## Cấu trúc

| Tệp | Chức năng |
|---|---|
| `baldseg/core.py` | CLAHE, polygon → class map, tính R |
| `baldseg/data.py` | Tải Roboflow, lọc trùng (dHash), chia split theo nhóm |
| `baldseg/model.py` | SegFormer 3 lớp (nền, tóc, hói): dataset, loss (CE có trọng số biên + Dice), dự đoán (TTA lật ngang + morphology) |
| `baldseg/yolo.py` | YOLO11-seg: instance mask → class map (instance tin cậy hơn ghi đè) |
| `baldseg/chart.py` | Đọc ngưỡng, R → mức |
| `baldseg/norwood_art.py` | Hình minh họa mức nhìn từ đỉnh đầu |
| `baldseg/metrics.py` | Dice/IoU/Precision/Recall; Accuracy, ±1, MAE, F1, kappa, confusion matrix |
| `scripts/prepare_data.py` | Tập phân đoạn `fifa-eg5yw/hairloss-segmentation-lf62l` → `data/` (80/10/10) |
| `scripts/prepare_topdown.py` | Ảnh Top-Down của `uze/hair-loss-nq8hh` (nhãn Norwood 2–7) → `data_topdown/`, chia theo người |
| `scripts/chart_thresholds.py` | Đo R trên ảnh bảng Hamilton-Norwood → `weights/norwood_thresholds.json` |
| `scripts/train_segformer.py` | Huấn luyện SegFormer-B2 → `weights/segformer.pt` |
| `scripts/train_yolo.py` | Fine-tune YOLO11m-seg (COCO) → `weights/yolo.pt` |
| `scripts/evaluate.py` | So sánh hai mô hình: phân đoạn trên tập test/valid, phân mức trên ảnh Top-Down có nhãn → `results/metrics.json` |
| `scripts/fetch_examples.py` | Ảnh mẫu đỉnh đầu từ tập test → `examples/` |
| `app.py` | Demo Gradio, chọn SegFormer hoặc YOLO |
| `scripts/vast_launch.sh`, `scripts/vast_run.sh` | Chạy toàn bộ trên Vast.ai: chuẩn bị dữ liệu, train, đánh giá, đóng gói, demo |

## Chạy

```bash
pip install -r requirements.txt
export ROBOFLOW_API_KEY=<key>
python -m scripts.prepare_data
python -m scripts.prepare_topdown
python -m scripts.train_segformer
python -m scripts.train_yolo
python -m scripts.evaluate
python -m scripts.fetch_examples
python app.py
```

Ngưỡng đã có sẵn trong `weights/norwood_thresholds.json`; tính lại bằng `python -m scripts.chart_thresholds <ảnh bảng Hamilton-Norwood>`.

Trên Vast.ai:

```bash
ROBOFLOW_API_KEY=<key> scripts/vast_launch.sh <OFFER_ID>
```

## Ngưỡng theo thang Hamilton-Norwood

R đo trên hình nhìn từ trên xuống của từng mức trong bảng, chỉ vùng sọ (từ đỉnh tới mép dưới tai, bỏ tai và mặt); ngưỡng là trung điểm giữa hai mức liền kề. Stage 2 và 3 có R gần nhau (Stage 3 lõm sâu ở giữa nhưng ít hói ở thái dương) nên gộp một khoảng.

| Mức | 1 | 2–3 | 3 Vertex | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| R trên bảng | 10% | 15–21% | 28% | 40% | 50% | 76% | 90% |
| Khoảng R | < 14% | 14–23% | 23–34% | 34–45% | 45–63% | 63–83% | ≥ 83% |

Khi đánh giá với nhãn 6 mức của tập Norwood (1–2, 3, 4, 5, 6, 7): Stage 1 → 1–2, khoảng 2–3 tách tại R = 18%, 3 Vertex → 3.

## Kết quả

Xem `results/metrics.json`. Hạn chế đã biết: ảnh đỉnh đầu ít thấy vành tóc hai bên/sau nên R ở mức nặng thường cao hơn hình vẽ khoảng một mức; tóc thưa lan tỏa vẫn bị tính là tóc; đường ngôi có thể bị tính là hói; trán nằm ngoài khung làm Stage 3 dễ ra Stage 1.
