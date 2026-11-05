# Đề xuất kỹ thuật: Phân đoạn và đánh giá mức độ rụng tóc kiểu nam từ ảnh

## 1. Bài toán

### 1.1. Mục tiêu

Xây dựng hệ thống học sâu nhận ảnh chụp đầu người (nam), tự động:

1. **Phân đoạn** vùng rụng tóc / da đầu lộ trên ảnh.
2. **Định lượng** mức độ rụng tóc bằng chỉ số diện tích. (có thể có hoặc không)
3. **Phân loại** mức độ rụng tóc theo một thang phân mức phù hợp với dữ liệu.

Kèm chương trình demo có giao diện: nhập ảnh, hiển thị vùng phân đoạn, tỷ lệ diện tích rụng tóc và mức độ dự đoán.

### 1.2. Đầu vào / đầu ra

|  | Mô tả |
|---|---|
| Đầu vào | Ảnh RGB chụp đỉnh đầu / đường chân tóc / phía sau đầu |
| Đầu ra 1 | Mask 2 lớp: `hair` (vùng có tóc), `bald_scalp` (vùng da đầu lộ do rụng tóc) |
| Đầu ra 2 | Chỉ số định lượng `R = S_bald / (S_bald + S_hair)` (optional) |
| Đầu ra 3 | Mức độ rụng tóc (4 mức, xem 1.3) |

### 1.3. Độ đo đánh giá

| Bài toán | Độ đo |
|---|---|
| Phân đoạn | Dice, IoU, Precision, Recall, F1 (theo lớp và trung bình) |
| Định lượng | MAE của R, Spearman (R dự đoán vs mức người chấm), ICC test–retest |
| Phân mức | Accuracy, Precision / Recall / F1 (macro), ma trận nhầm lẫn |

## 2. Tập dữ liệu

### 2.1. Nguồn

```
hairloss-segmentation
```

| Thuộc tính | Giá trị |
|---|---|
| Số ảnh (version 2) | 1.251 (train 1.186 / valid 42 / test 23) |
| Ảnh gốc thật | ~461 (phần còn lại là bản xoay 90° do Roboflow tạo) |
| Định dạng | YOLOv8 instance segmentation (polygon) |
| Lớp | `bald` , `hair` |
| Kích thước | resize 640×640 (stretch), auto-orient |
| Giấy phép | MIT |
| Label rỗng | 0 |

### 2.2. Hạn chế của tập và cách xử lý

| Hạn chế | Xử lý |
|---|---|
| Chỉ 461 ảnh gốc | Gộp thêm dữ liệu |
| Nhãn do cộng đồng, không kiểm chứng lâm sàng | Tự gán lại ~150 ảnh test có bác sĩ da liễu review làm gold standard |
| Augment xoay 90° bake sẵn trong train | Chia lại theo `source_id` ; các bản xoay của cùng ảnh gốc nằm cùng split |
| Không có nhãn mức độ | Suy từ R + 2 người chấm nhãn mức trên tập đánh giá (đo kappa) |
| Thiếu ảnh đường chân tóc phía trước | Bổ sung từ tập `hair-loss-nq8hh` (có view Front) và ảnh tự chụp |

## 3. Hai phương án huấn luyện

### 3.1. Phương án A : Đơn giản (baseline)

**Kiến trúc:** một mô hình phân đoạn có sẵn + hậu xử lý bằng quy tắc.

```
Ảnh → YOLOv8-seg / YOLO11-seg (pretrained COCO, fine-tune)
    → mask {hair, bald_scalp}
    → đếm pixel → R = S_bald / (S_bald + S_hair)
    → ngưỡng hóa R (Jenks/quantile học từ train) → mức 0..3
```

|  |  |
|---|---|
| Phân đoạn | Ultralytics YOLOv8-seg hoặc YOLO11-seg, fine-tune từ trọng số COCO, train bằng CLI |
| Định lượng | Đếm pixel trên mask dự đoán |
| Phân mức | Không có mô hình riêng : chỉ ngưỡng hóa R (có thể thêm 1 Decision Tree nông trên {R, độ lùi chân tóc}) |
| Tiền xử lý | Resize 640, auto-orient, CLAHE |
| Loss / tối ưu | Mặc định của Ultralytics (seg loss) |
| Phần cứng | 1 GPU phổ thông (T4 / RTX 3060), hoặc Colab free |
| Triển khai | Xuất ONNX / TensorRT, app Gradio |

**Ưu điểm:** nhanh (train vài giờ), rất ít code, dễ tái lập, dễ deploy, đủ để có baseline và sản phẩm demo hoạt động.

**Nhược điểm:** biên mask của YOLO-seg khá thô ở vùng chuyển tiếp tóc–da đầu; không tận dụng được ảnh không có nhãn; chất lượng phân mức phụ thuộc hoàn toàn vào phân đoạn; không có ước lượng độ tin cậy.

**Kết quả kỳ vọng:** Dice ~0.80–0.87 (lớp `bald` ), Accuracy phân mức 4 lớp ~0.75–0.83.

### 3.2. Phương án B : Phức tạp (đa nhiệm, bán giám sát)

**Kiến trúc:** một backbone chung, nhiều đầu ra, kết hợp dữ liệu không nhãn.

```
                         ┌─ Head phân đoạn  → mask {hair, bald_scalp}
Ảnh → Encoder mạnh ──────┤
   (SegFormer-B2 /       ├─ Head hồi quy    → R (giá trị liên tục)
    Mask2Former /        │
    U-Net++ w/ SAM enc.) └─ Head phân mức   → mức 0..3
```

| Thành phần | Chi tiết |
|---|---|
| Encoder | SegFormer-B2 hoặc Swin-Tiny, pretrained ImageNet; hoặc dùng image encoder của SAM (đóng băng một phần) |
| Pretrain phân đoạn tóc | Huấn luyện sơ bộ trên CelebAMask-HQ / Figaro-1k (lớp `hair` ) trước khi fine-tune |
| Loss phân đoạn | Dice + BCE + boundary loss (nhấn biên) |
| Loss đa nhiệm | `L = L_seg + λ1·L_R (Smooth-L1) + λ2·L_grade (CE có trọng số lớp)` ; R được tính lại từ mask dự đoán và ràng buộc nhất quán với head hồi quy |
| Bán giám sát | Pseudo-label ảnh từ `hair-loss-nq8hh` (không có mask) bằng mô hình đã train, lọc theo độ tin cậy, đưa lại vào train (self-training) |
| Ước lượng bất định | MC Dropout hoặc test-time augmentation → bản đồ độ tin cậy, đặc biệt cho vùng thưa mờ |
| Hậu xử lý | CRF / morphology làm mượt biên; lọc thành phần liên thông nhỏ |
| Phần cứng | 1 GPU 12–16 GB (RTX 3080/4070, A10, hoặc RunPod) |

**Ưu điểm:** biên chính xác hơn; phân mức ổn định hơn nhờ học chung đặc trưng và ràng buộc nhất quán R; tận dụng được ảnh không nhãn; có bản đồ độ tin cậy để giải thích và cảnh báo ca khó.

**Nhược điểm:** nhiều code và cấu hình; thời gian tinh chỉnh siêu tham số lâu; cần GPU khá; rủi ro overfit với dữ liệu gốc nhỏ nếu không kiểm soát tốt.

**Kết quả kỳ vọng:** Dice ~0.86–0.92 (lớp `bald` ), Accuracy phân mức 4 lớp ~0.83–0.90.

### 3.3. So sánh nhanh

| Tiêu chí | Phương án A | Phương án B |
|---|---|---|
| Độ phức tạp code | Thấp | Cao |
| Thời gian train | Vài giờ | 1–2 ngày (gồm tinh chỉnh) |
| Phần cứng | GPU phổ thông / Colab | GPU 12–16 GB |
| Chất lượng biên mask | Trung bình | Tốt |
| Tận dụng dữ liệu không nhãn | Không | Có |
| Ước lượng độ tin cậy | Không | Có |
| Vai trò | Baseline + demo | Kết quả chính của đồ án |

## 4. Sản phẩm bàn giao

| # | Sản phẩm |
|---|---|
| 1 | Báo cáo tổng quan: rụng tóc kiểu nam, các thang đánh giá, các phương pháp học sâu liên quan |
| 2 | Mô tả tập dữ liệu + quy trình thu thập / tiền xử lý / gán nhãn + tập dữ liệu kèm nhãn |
| 3 | Mô hình phân đoạn (Phương án A và B) + trọng số đã huấn luyện |
| 4 | Mô-đun tính chỉ số định lượng R và ánh xạ ra mức độ |
| 5 | Kết quả đánh giá: Dice/IoU/Precision/Recall/F1 (phân đoạn); Accuracy/P/R/F1 + confusion matrix (phân mức); MAE của R |
| 6 | Chương trình demo (Gradio/Streamlit): nhập ảnh → mask + R + mức độ |
| 7 | Mã nguồn đầy đủ, tài liệu chạy lại, báo cáo đồ án |

## 5. Khối lượng công việc và tiến độ

| # | Hạng mục | Nội dung |
|---|---|---|
| 1 | Chuẩn bị dữ liệu | Lọc/dedupe, chia lại theo `source_id` , tính R, gán nhãn mức, tự gán ~150 ảnh test (2 người + review) |
| 2 | Phương án A | YOLO-seg fine-tune, mô-đun R, ngưỡng hóa phân mức, đánh giá |
| 3 | Phương án B | Đa nhiệm SegFormer/Mask2Former, pretrain tóc, bán giám sát, bất định, hậu xử lý, tinh chỉnh |
| 4 | Đánh giá | Toàn bộ độ đo, biểu đồ, phân tích lỗi, so sánh A/B/baseline |
| 5 | App demo | Gradio: upload, overlay mask, hiển thị R và mức, đóng gói |
| 6 | Báo cáo | Tổng quan, phương pháp, thực nghiệm, kết luận |
|  | **Tổng** |  |

**Tiến độ dự kiến:** 8 – 10 tuần (làm bán thời gian), hoặc 5 – 6 tuần (toàn thời gian).

| Tuần | Mốc |
|---|---|
| 1 – 2 | Hạng mục 1, hoàn tất tập dữ liệu + nhãn |
| 2 – 3 | Hạng mục 2 (Phương án A) + app demo v1 |
| 3 – 5 | Hạng mục 3 (Phương án B) |
| 5 – 6 | Hạng mục 4 (đánh giá) |
| 5 – 6 | Hạng mục 6 (báo cáo), hoàn thiện |

## 6. Chi phí sản phẩm

| Hạng mục | Chi phí | Nội dung |
|---|---:|---|
| Sản phẩm + mã nguồn | **6.000.000 VNĐ** | Bao gồm mã nguồn đầy đủ, mã nguồn huấn luyện, chi phí thuê server/GPU phục vụ huấn luyện và các thành phần kỹ thuật liên quan |
| Báo cáo đồ án | **1.000.000 VNĐ** | Báo cáo đồ án theo nội dung và phạm vi của đề tài |
| Phụ thu báo cáo > 60 trang | **+500.000 VNĐ** | Áp dụng khi yêu cầu báo cáo có độ dài trên 60 trang |
| **Tổng cơ bản** | **7.000.000 VNĐ** | Sản phẩm + mã nguồn + chi phí huấn luyện + báo cáo ≤ 60 trang |

Đối với gói **sản phẩm + mã nguồn**, khách hàng thực hiện **đặt cọc 30%** tổng chi phí trước khi bắt đầu triển khai. Sau khi hoàn thành và bàn giao **demo** để khách hàng kiểm tra, khách hàng thanh toán **70% chi phí còn lại**. Sau khi hoàn tất thanh toán, tiến hành chuyển giao đầy đủ sản phẩm, mã nguồn, mô hình, tài liệu hướng dẫn và các thành phần thuộc phạm vi đã thống nhất.

Đối với trường hợp thực hiện **báo cáo theo tuần**, nội dung công việc sẽ được chia theo từng tuần. Mỗi tuần sẽ thống nhất trước các nội dung cần hoàn thiện, sau đó tiến hành thực hiện và bàn giao phần nội dung đã hoàn thành trong tuần. Chi phí sẽ được thanh toán tương ứng với **phần nội dung đã thực hiện và hoàn thiện trong tuần đó** trước khi tiếp tục sang nội dung của tuần tiếp theo.

- **Sản phẩm + mã nguồn:** Cọc 30% → hoàn thành demo → thanh toán 70% còn lại → bàn giao đầy đủ.
- **Báo cáo theo tuần:** Thống nhất nội dung → thực hiện → bàn giao nội dung trong tuần → thanh toán chi phí phần đã hoàn thành → tiếp tục tuần tiếp theo.

> **Lưu ý:** Các yêu cầu phát sinh ngoài phạm vi nội dung đã thống nhất ban đầu hoặc yêu cầu thay đổi lớn sẽ được trao đổi và báo giá riêng trước khi thực hiện.
