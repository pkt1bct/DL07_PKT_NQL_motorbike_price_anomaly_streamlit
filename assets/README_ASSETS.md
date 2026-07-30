# assets/

Thư mục này chứa toàn bộ tài nguyên giao diện (UI Assets) của ứng dụng Streamlit.

---

## Cấu trúc

```text
assets/

custom.css
README_ASSETS.md

logo.png                 (khuyến nghị)
favicon.png              (khuyến nghị)
banner.png               (khuyến nghị)
placeholder_motorbike.png

icons/
    anomaly.png
    prediction.png
    dashboard.png
    upload.png
```

---

## Vai trò

Các file trong thư mục assets chỉ phục vụ giao diện người dùng.

Không ảnh hưởng đến:

- Predictor
- Feature Engineering
- Preprocess
- Anomaly Detection
- Recommendation
- Visualization
- các model .pkl

Do đó việc thay đổi CSS sẽ không làm thay đổi kết quả dự đoán.

---

## custom.css

File này dùng để:

- tùy chỉnh giao diện Streamlit
- màu sắc
- sidebar
- button
- metric
- table
- card
- responsive layout

Để sử dụng:

```python
from pathlib import Path

css_file = Path("assets/custom.css")

if css_file.exists():
    with open(css_file, encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True,
        )
```

---

## Logo

Có thể thêm:

```
assets/logo.png
```

và sử dụng

```python
st.logo("assets/logo.png")
```

hoặc

```python
st.image("assets/logo.png")
```

---

## Banner

Nếu có:

```
assets/banner.png
```

có thể hiển thị

```python
st.image(
    "assets/banner.png",
    use_container_width=True,
)
```

---

## Placeholder

```
assets/placeholder_motorbike.png
```

dùng khi:

- người dùng chưa upload dữ liệu
- chưa có ảnh minh họa

---

## Icons

Khuyến nghị:

```
icons/dashboard.png

icons/prediction.png

icons/anomaly.png

icons/upload.png
```

để minh họa:

- Dashboard
- Prediction
- Anomaly Detection
- Batch Upload

---

## Tương thích

Đã kiểm tra tương thích với:

- app.py
- preprocess.py
- feature_engineering.py
- predictor.py
- anomaly_detector.py
- recommendation.py
- visualization.py

và:

- model_phothong.pkl
- model_trungcap.pkl
- model_caocap.pkl

---

## Khuyến nghị

Giữ nguyên cấu trúc project:

```text
motorbike_price_anomaly_streamlit/

app.py

requirements.txt

setup.sh

Procfile

assets/

src/

models/

data/
```

Điều này giúp ứng dụng triển khai ổn định trên:

- Streamlit Community Cloud
- Render
- Railway
- Heroku (Python Buildpack)

mà không cần sửa đường dẫn trong mã nguồn.