# Bald segmentation

Phân đoạn vùng tóc / da đầu lộ, tính tỷ lệ rụng tóc `R = S_bald / (S_bald + S_hair)` và phân 4 mức độ. Triển khai Phương án A trong `report.md`: YOLO11-seg fine-tune + ngưỡng hóa R.

## Cấu trúc

| Tệp | Chức năng |
|---|---|
| `baldseg/core.py` | CLAHE, chuyển polygon / kết quả YOLO thành class map, tính R, ngưỡng phân mức |
| `baldseg/metrics.py` | Dice, IoU, Precision, Recall theo lớp; Accuracy, P/R/F1 macro, kappa, confusion matrix; MAE, Spearman, ICC |
| `scripts/prepare_data.py` | Tải dataset Roboflow, loại ảnh trùng (dHash), chia lại 80/10/10 theo `source_id`, áp CLAHE |
| `scripts/train.py` | Fine-tune YOLO11-seg, lưu `weights/best.pt` và ngưỡng mức `weights/grades.json` |
| `scripts/evaluate.py` | Đánh giá trên tập test/valid, lưu `results/metrics_<split>.json` |
| `scripts/vast_onstart.sh` | Script chạy toàn bộ pipeline trên instance Vast.ai |
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
python app.py --weights weights
```

## Huấn luyện trên Vast.ai

```bash
vastai create instance <OFFER_ID> --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime --disk 30 --ssh \
  --env "-e ROBOFLOW_API_KEY=<key> -e BRANCH=main" --onstart scripts/vast_onstart.sh
vastai logs <INSTANCE_ID> --tail 100
vastai copy <INSTANCE_ID>:/workspace/result.tgz ./
```

## Ghi chú phương pháp

- Dataset: Roboflow `fifa-eg5yw/hairloss-segmentation-lf62l` version 1, 461 ảnh gốc, lớp `bald`, `hair`, resize 640×640, không augment sẵn. Augment xoay/lật thực hiện lúc train nên không rò rỉ giữa các split.
- Mask lớp: chồng `bald` lên `hair` (nhãn); với dự đoán, instance có độ tin cậy cao hơn ghi đè.
- Mức độ: không có nhãn mức, nên mức tham chiếu suy từ R của nhãn gốc; ngưỡng là các phân vị 25/50/75 của R trên tập train (`--grade-method kmeans` cho phân cụm 1D).
- F1 pixel trùng Dice. ICC test–retest đo trên ảnh gốc và ảnh lật ngang + đổi sáng.
