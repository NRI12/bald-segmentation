# Bald segmentation

Phân đoạn vùng tóc / da đầu lộ, tính tỷ lệ rụng tóc `R = S_bald / (S_bald + S_hair)` và phân 4 mức độ. Triển khai theo `report.md`: Phương án A (YOLO11-seg + phân mức bằng cây quyết định) và Phương án B (SegFormer-B2 đa nhiệm, bán giám sát, MC Dropout).

## Cấu trúc

| Tệp | Chức năng |
|---|---|
| `baldseg/core.py` | CLAHE, chuyển polygon / kết quả YOLO thành class map, tính R, ngưỡng phân mức |
| `baldseg/data.py` | Tải Roboflow, dHash dedupe, chia split theo nhóm |
| `baldseg/multitask.py` | Phương án B: mô hình SegFormer 3 đầu ra, dataset, loss, MC Dropout + TTA, hậu xử lý morphology |
| `baldseg/metrics.py` | Dice, IoU, Precision, Recall theo lớp; Accuracy, P/R/F1 macro, kappa, confusion matrix; MAE, Spearman, ICC |
| `scripts/prepare_data.py` | Tải dataset Roboflow, loại ảnh trùng (dHash), chia lại 80/10/10 theo `source_id`, áp CLAHE |
| `scripts/train.py` | Fine-tune YOLO11-seg, lưu `weights/best.pt` và ngưỡng mức `weights/grades.json` |
| `scripts/evaluate.py` | Đánh giá Phương án A trên tập test/valid, lưu `results/metrics_<split>.json` |
| `scripts/prepare_grades.py` | Tải `uze/hair-loss-nq8hh` (nhãn Norwood 2–7), chia theo người, ánh xạ 4 mức |
| `scripts/fit_grader.py` | Phương án A: cây quyết định trên {R, diện tích hói, diện tích tóc, góc chụp}, chọn độ sâu theo kappa valid |
| `scripts/train_b.py` | Phương án B: huấn luyện đa nhiệm (seg + R + mức) |
| `scripts/pseudo_label.py` | Sinh pseudo-mask cho ảnh Norwood theo ngưỡng độ tin cậy |
| `scripts/evaluate_b.py` | Đánh giá Phương án B: phân đoạn, R, bất định, phân mức theo nhãn Norwood |
| `scripts/vast_launch.sh` | Đóng gói mã nguồn vào onstart và tạo instance Vast.ai |
| `scripts/cv_grades.py` | Đánh giá chéo 5-fold theo người, liệt kê nhãn nghi sai |
| `scripts/review_suspects.py` | Ảnh ghép 5 góc cho từng người nghi sai nhãn + CSV để sửa nhãn |
| `scripts/vast_cv.sh` | Chạy đánh giá chéo trên Vast |
| `scripts/vast_norwood.sh` | Fine-tune 6 stage Norwood trên Vast rồi chạy demo |
| `baldseg/face.py` | Face-parsing để loại mặt, tai, cổ khỏi mask tóc/hói |
| `scripts/vast_face.sh` | Fit lại cây A với mask đã lọc mặt, đánh giá, chạy demo |
| `baldseg/norwood_art.py` | Vẽ hình minh họa từng stage (nhìn từ đỉnh đầu) |
| `scripts/vast_onstart.sh` | Pipeline chạy trên instance: chuẩn bị dữ liệu, train, đánh giá |
| `app.py` | Demo Gradio: 1–5 ảnh/người theo góc chụp, gộp xác suất → stage Norwood, minh họa, mask, bản đồ bất định |
| `examples/` | Ảnh mẫu từ tập test: `seg*` (hairloss-segmentation, MIT), `norwood*` (uze/hair-loss-nq8hh, CC BY 4.0) |
| `scripts/vast_demo.sh` | Chạy demo trên Vast + Cloudflare quick tunnel |

## Cài đặt

```bash
pip install -r requirements.txt
export ROBOFLOW_API_KEY=<key>
```

## Chạy

```bash
python -m scripts.prepare_data
python -m scripts.train --model yolo11m-seg.pt --epochs 200
python -m scripts.evaluate --split test
python -m scripts.prepare_grades
python -m scripts.fit_grader
python app.py --plan a
```

Phương án B:

```bash
python -m scripts.train_b --epochs 40 --out weights_b/stage1.pt
python -m scripts.pseudo_label
python -m scripts.train_b --init weights_b/stage1.pt --pseudo data_grade/pseudo --epochs 20 --lr 3e-5 --out weights_b/stage2.pt
python -m scripts.evaluate_b
python app.py --plan b
```

## Huấn luyện trên Vast.ai

```bash
ROBOFLOW_API_KEY=<key> scripts/vast_launch.sh <OFFER_ID>
ROBOFLOW_API_KEY=<key> ONSTART=scripts/vast_demo.sh BUNDLE="app.py baldseg scripts/fetch_examples.py requirements.txt" scripts/vast_launch.sh <OFFER_ID>
vastai copy <TRAIN_ID>:/workspace/result.tgz <DEMO_ID>:/workspace/
vastai logs <INSTANCE_ID> --tail 100
vastai copy <INSTANCE_ID>:/workspace/result.tgz ./
```

## Kết quả Phương án A ban đầu (YOLO11m-seg, 200 epoch, RTX 3060, mức suy từ R)

| Split | Dice bald | IoU bald | Dice hair | Dice TB | MAE R | Spearman R–mức | ICC | Acc mức | F1 macro | Kappa (quadratic) |
|---|---|---|---|---|---|---|---|---|---|---|
| valid (46) | 0.815 | 0.687 | 0.855 | 0.835 | 0.103 | 0.70 | 0.996 | 0.52 | 0.37 | 0.59 |
| test (46) | 0.717 | 0.559 | 0.878 | 0.798 | 0.116 | 0.42 | 0.996 | 0.50 | 0.42 | 0.40 |

Ngưỡng R (phân vị train): 0.017 / 0.087 / 0.358. Mask mAP50 (valid) 0.743.

Phân đoạn và độ ổn định test–retest đạt yêu cầu baseline. Phân mức còn yếu ở mức 1–2: các ngưỡng thấp rất sát nhau nên vùng hói nhỏ bị bỏ sót sẽ đẩy ảnh về mức 0. Mức 3 nhận đúng 8/12 (test) và 18/21 (valid).

## Phân mức Phương án A theo nhãn Norwood (test 190 ảnh, 4 mức)

| Cách phân mức | Accuracy | F1 macro | Kappa (quadratic) |
|---|---|---|---|
| Ngưỡng R phân vị (cũ) | 0.395 | 0.264 | 0.311 |
| Cây quyết định độ sâu 4 trên {R, diện tích hói, diện tích tóc, góc chụp} | 0.384 | 0.364 | 0.486 |

Accuracy của cây theo góc chụp: Top-Down 0.63, Front 0.53, Back 0.32, Right 0.27, Left 0.18. Cây cải thiện rõ kappa và F1 (không còn dồn về một mức), nhưng ảnh nghiêng/sau đầu khác phân phối với tập phân đoạn nên R kém tin cậy ở các góc này. Lần chạy lại này: Dice bald test 0.742, valid 0.809.

## So sánh Phương án A và B (test)

Phân đoạn trên 46 ảnh test của tập phân đoạn; phân mức trên 190 ảnh test có nhãn Norwood.

| Độ đo | A: YOLO11m-seg + cây | B: SegFormer-B2 đa nhiệm |
|---|---|---|
| Dice bald | 0.742 | **0.812** |
| IoU bald | 0.590 | **0.684** |
| Dice hair | 0.878 | **0.902** |
| Dice trung bình | 0.810 | **0.857** |
| MAE của R | 0.111 | **0.067** (mask), 0.083 (head) |
| ICC test–retest | 0.999 | 1.000 |
| Accuracy mức (Norwood) | 0.384 | **0.547** |
| F1 macro mức | 0.364 | **0.502** |
| Kappa (quadratic) mức | 0.486 | **0.711** |
| Spearman bất định – lỗi pixel | – | 0.894 |

Valid của B: Dice bald 0.875, Dice TB 0.883, Accuracy mức 0.596, kappa 0.626.

Accuracy mức của B theo góc chụp (test): Top-Down 0.68, Left 0.56, Front 0.56, Back 0.47, Right 0.46. So với A, B cải thiện mạnh ở ảnh nghiêng và sau đầu nhờ đầu phân mức học trực tiếp từ nhãn Norwood.

Ma trận nhầm lẫn mức của B (test, hàng = thật):

```
[[15  6  0  0]
 [15 12 16  0]
 [ 5  6 20  9]
 [ 2  5 22 57]]
```

Lỗi chủ yếu nằm giữa hai mức liền kề (đúng tính thứ bậc, kappa 0.71). Bản đồ bất định tương quan mạnh với lỗi phân đoạn (Spearman 0.89), dùng được để cảnh báo ca khó. Pseudo-label: giữ 844/908 ảnh Norwood ở vòng 2.

## Phân loại theo thang Hamilton-Norwood (6 stage, phiên bản hiện tại)

Đầu ra mức độ dùng trực tiếp thang Norwood: Stage 1-2, 3, 4, 5, 6, 7 (tập nhãn không có Stage 1 nên gộp với 2). Stage 3 Vertex là cờ phụ theo luật: ảnh Top-Down, dự đoán Stage 3 và có vùng hói ở đỉnh được tóc bao quanh. B fine-tune 15 epoch từ `stage2.pt` với đầu phân mức 6 lớp, loss CE có trọng số + kỳ vọng thứ bậc (SmoothL1); A fit lại cây quyết định 6 lớp. Demo hiển thị stage, xác suất từng stage và hình minh họa (`baldseg/norwood_art.py`).

| Test (190 ảnh Norwood) | A: YOLO + cây | B: SegFormer đa nhiệm |
|---|---|---|
| Accuracy đúng stage | 0.289 | **0.489** |
| Accuracy ±1 stage | 0.668 | **0.858** |
| MAE (stage) | 1.16 | **0.71** |
| F1 macro | 0.270 | **0.481** |
| Kappa (quadratic) | 0.568 | **0.784** |

Valid của B: accuracy 0.346, ±1 stage 0.883, kappa 0.745. Phân đoạn của B sau fine-tune giữ nguyên: Dice bald test 0.815, valid 0.877; MAE của R 0.066.

## Đánh giá chéo 5-fold theo người (Phương án B)

Tập test cũ chỉ có 20 người nên số đo dao động mạnh. `scripts/cv_grades.py` chia 133 người (1.286 ảnh) thành 5 fold phân tầng theo stage, mỗi fold train lại từ đầu (SegFormer-B2, 384 px, 12 epoch, không pseudo-label) và dự đoán trên người chưa thấy. Mức người = trung bình xác suất các góc chụp.

| | Đúng stage | ±1 stage | MAE (stage) | F1 macro | Kappa |
|---|---|---|---|---|---|
| Theo ảnh (1.286) | 0.409 | 0.844 | 0.78 | 0.423 | 0.773 |
| Theo người (133) | 0.414 | 0.880 | 0.74 | 0.430 | 0.789 |

Accuracy theo góc (ảnh): Top-Down 0.48, Front 0.41, Left 0.41, Right 0.39, Back 0.36. Accuracy từng fold: 0.39 / 0.43 / 0.41 / 0.37 / 0.46.

Kết luận: mô hình xếp đúng thứ bậc (kappa ~0.79, 88% người lệch tối đa 1 stage) nhưng đúng chính xác stage chỉ ~41%; ranh giới giữa các stage liền kề là chỗ khó nhất. 46 người có dự đoán lệch nhãn ≥2 stage hoặc lệch với độ tin cậy ≥60% được liệt kê để rà soát nhãn: `python -m scripts.review_suspects` tạo ảnh ghép 5 góc và `results/cv/suspects/suspects.csv` (cột `corrected_norwood` để điền nhãn đã sửa). Rà soát nhanh cho thấy danh sách gồm cả lỗi mô hình (ví dụ tóc rất ngắn, da sẫm bị đoán quá nặng) lẫn nhãn có thể sai hoặc nằm ở ranh giới.

## Loại vùng mặt, tai, cổ khỏi mask

Tập phân đoạn chỉ có 2 lớp `hair`/`bald` và gần như không có ảnh nghiêng, nên mọi vùng da (má, tai, cổ) đều bị dự đoán là `bald`, làm R bị thổi phồng ở ảnh Front/Left/Right. `baldseg/face.py` dùng mô hình face-parsing có sẵn (`jonathandinu/face-parsing`, SegFormer trên CelebAMask-HQ) làm hậu xử lý, không cần train lại:

- luôn loại: mắt, lông mày, kính, mũi, miệng, môi, tai, cổ, áo;
- loại thêm vùng da/râu phía dưới đường lông mày (hướng "xuống" = lông mày → mũi/miệng/cổ, nên không phụ thuộc chiều xoay ảnh), giới hạn trong bao lồi của các bộ phận mặt; trán và da đầu được giữ lại;
- không dùng thẳng lớp `skin` của face-parsing vì mô hình này gán cả da đầu hói là `skin`.

Ảnh hưởng (test):

| | Không lọc mặt | Có lọc mặt |
|---|---|---|
| A: Dice bald (phân đoạn, 46 ảnh) | 0.742 | 0.745 |
| B: Dice bald / MAE R (46 ảnh) | 0.815 / 0.066 | 0.817 / 0.066 |
| A: cây 6 stage, accuracy / ±1 / kappa (190 ảnh) | 0.289 / 0.668 / 0.568 | 0.247 / 0.742 / 0.622 |

Ảnh test của tập phân đoạn chủ yếu là đỉnh đầu nên Dice gần như không đổi; tác dụng chính là R ở ảnh mặt/nghiêng không còn bị tính cả khuôn mặt. Đầu phân mức của B không dùng mask nên stage của B không đổi. Hạn chế: ở ảnh sau đầu, phần gáy/cổ sát chân tóc đôi khi vẫn bị tính là `bald` vì face-parsing không nhận ra cổ từ phía sau.

## Ghi chú phương pháp

- Dataset: Roboflow `fifa-eg5yw/hairloss-segmentation-lf62l` version 1, 461 ảnh gốc, lớp `bald`, `hair`, resize 640×640, không augment sẵn. Augment xoay/lật thực hiện lúc train nên không rò rỉ giữa các split.
- Mask lớp: chồng `bald` lên `hair` (nhãn); với dự đoán, instance có độ tin cậy cao hơn ghi đè.
- Mức độ: không có nhãn mức, nên mức tham chiếu suy từ R của nhãn gốc; ngưỡng là các phân vị 25/50/75 của R trên tập train (`--grade-method kmeans` cho phân cụm 1D).
- F1 pixel trùng Dice. ICC test–retest đo trên ảnh gốc và ảnh lật ngang + đổi sáng.
- Nhãn mức thật lấy từ `uze/hair-loss-nq8hh` (CC BY 4.0, 5 góc chụp/người): Phiên bản 4 mức trước đó: Norwood 2 → 0, 3 → 1, 4–5 → 2, 6–7 → 3; hiện tại dùng 6 stage (xem mục trên). Chia split theo người để không rò rỉ.
- Phương án B: loss `L = L_seg (CE có trọng số biên + Dice) + 1.0·SmoothL1(R) + 0.5·CE(mức, trọng số lớp) + 0.2·nhất quán(R head, R mềm từ mask)`. Vòng 1 học trên mask thật + nhãn mức; vòng 2 thêm pseudo-mask (pixel conf ≥ 0.9, ảnh giữ lại nếu ≥ 60% pixel đủ tin cậy). Bất định: 8 lượt MC Dropout × lật ngang, entropy chuẩn hóa.
- Chưa làm: pretrain trên CelebAMask-HQ / Figaro-1k và CRF (thay bằng morphology + lọc thành phần liên thông nhỏ).
