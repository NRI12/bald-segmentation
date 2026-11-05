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
| `scripts/vast_onstart.sh` | Pipeline chạy trên instance: chuẩn bị dữ liệu, train, đánh giá |
| `app.py` | Demo Gradio: ảnh → overlay mask, R, mức độ |

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

## Ghi chú phương pháp

- Dataset: Roboflow `fifa-eg5yw/hairloss-segmentation-lf62l` version 1, 461 ảnh gốc, lớp `bald`, `hair`, resize 640×640, không augment sẵn. Augment xoay/lật thực hiện lúc train nên không rò rỉ giữa các split.
- Mask lớp: chồng `bald` lên `hair` (nhãn); với dự đoán, instance có độ tin cậy cao hơn ghi đè.
- Mức độ: không có nhãn mức, nên mức tham chiếu suy từ R của nhãn gốc; ngưỡng là các phân vị 25/50/75 của R trên tập train (`--grade-method kmeans` cho phân cụm 1D).
- F1 pixel trùng Dice. ICC test–retest đo trên ảnh gốc và ảnh lật ngang + đổi sáng.
- Nhãn mức thật lấy từ `uze/hair-loss-nq8hh` (CC BY 4.0, 5 góc chụp/người): Norwood 2 → 0, 3 → 1, 4–5 → 2, 6–7 → 3. Chia split theo người để không rò rỉ.
- Phương án B: loss `L = L_seg (CE có trọng số biên + Dice) + 1.0·SmoothL1(R) + 0.5·CE(mức, trọng số lớp) + 0.2·nhất quán(R head, R mềm từ mask)`. Vòng 1 học trên mask thật + nhãn mức; vòng 2 thêm pseudo-mask (pixel conf ≥ 0.9, ảnh giữ lại nếu ≥ 60% pixel đủ tin cậy). Bất định: 8 lượt MC Dropout × lật ngang, entropy chuẩn hóa.
- Chưa làm: pretrain trên CelebAMask-HQ / Figaro-1k và CRF (thay bằng morphology + lọc thành phần liên thông nhỏ).
