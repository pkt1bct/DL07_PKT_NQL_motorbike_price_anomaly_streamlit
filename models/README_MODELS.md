# Model và artifact cần đưa vào thư mục này

Theo file P2, mô hình dự đoán giá được huấn luyện riêng theo phân khúc và lưu dưới dạng pickle:

```text
models/model_phothong.pkl
models/model_trungcap.pkl
models/model_caocap.pkl
```

Để giao diện Streamlit hiển thị báo cáo đánh giá và phát hiện bất thường, bước xây dựng tiếp theo sẽ hỗ trợ thêm các artifact sau:

```text
models/model_metrics.json
models/anomaly_artifacts.pkl
```

Trong đó:

- `model_metrics.json`: MAE, RMSE, R² và tên mô hình theo từng phân khúc.
- `anomaly_artifacts.pkl`: ngưỡng, thống kê residual, phân vị và các mô hình phát hiện bất thường cần dùng khi dự đoán mới.

Nếu notebook P2 chưa xuất hai file artifact này, project sẽ bổ sung script trong `src/` để tạo chúng từ dữ liệu và model đã huấn luyện.

Không đổi tên ba file model chính, vì `app.py` sẽ tải đúng các đường dẫn trên.
