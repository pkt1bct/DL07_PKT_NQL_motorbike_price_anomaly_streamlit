# BƯỚC 2 — Đưa dữ liệu và model vào project

Cấu trúc sau bước 2:

```text
motorbike_price_anomaly_streamlit/
├── .streamlit/
├── assets/
├── data/
│   ├── data_motobikes.xlsx          # Người dùng bổ sung từ bài P2
│   ├── sample_batch_upload.csv      # Đã tạo
│   └── README_DATA.md               # Đã tạo
├── models/
│   ├── model_phothong.pkl           # Xuất từ notebook P2
│   ├── model_trungcap.pkl           # Xuất từ notebook P2
│   ├── model_caocap.pkl             # Xuất từ notebook P2
│   ├── model_metrics.json           # Sẽ tạo từ kết quả đánh giá
│   ├── anomaly_artifacts.pkl        # Sẽ tạo từ pipeline bất thường
│   └── README_MODELS.md             # Đã tạo
├── src/
├── README_STEP1.md
└── README_STEP2.md
```

## Các thao tác cần làm trên máy của bạn

1. Sao chép `data_motobikes.xlsx` vào thư mục `data/`.
2. Sao chép ba model được notebook P2 tạo ra vào thư mục `models/`:
   - `model_phothong.pkl`
   - `model_trungcap.pkl`
   - `model_caocap.pkl`
3. Giữ nguyên tên file để ứng dụng tải model chính xác.

Do các file dữ liệu/model thực tế chưa được đính kèm trong cuộc trò chuyện, bản project này chỉ tạo đúng vị trí, file mẫu và hướng dẫn. Bước 3 sẽ xây dựng `app.py`, các module xử lý đặc trưng, tải model, dự đoán giá, phát hiện bất thường, vẽ biểu đồ và đưa ra khuyến nghị.
