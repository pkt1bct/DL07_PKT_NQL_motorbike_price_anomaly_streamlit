# -*- coding: utf-8 -*-
"""
Streamlit GUI cho bài toán:
1) Dự đoán giá đề xuất xe máy cũ.
2) Phát hiện mức giá rao bất thường.
3) Phân tích và đưa ra khuyến nghị.

File này chỉ phụ trách giao diện và điều phối luồng xử lý.
Logic làm sạch dữ liệu, feature engineering, dự đoán, phát hiện bất thường,
trực quan hóa và khuyến nghị được đặt trong thư mục src/.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import base64
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# IMPORT CÁC MODULE CỦA PROJECT
# ---------------------------------------------------------------------
from src.preprocess import (
    load_dataset,
    prepare_eda_data,
    get_data_quality_summary,
    extract_district,
    validate_batch_columns,
    get_form_options,
)
from src.feature_engineering import (
    build_input_dataframe,
)
from src.predictor import (
    load_models,
    predict_batch,
    predict_price,
)
from src.anomaly_detector import (
    analyze_batch_anomalies,
    analyze_price_anomaly,
)
from src.visualization import (
    plot_anomaly_score_components,
    plot_brand_distribution,
    plot_correlation_heatmap,
    plot_district_distribution,
    plot_model_metrics,
    plot_prediction_comparison,
    plot_price_by_brand,
    plot_price_by_vehicle_type,
    plot_price_distribution,
    plot_segment_distribution,
    plot_year_price_scatter,
)
from src.recommendation import (
    generate_batch_recommendations,
    generate_recommendation,
)


# =====================================================================
# 1. CẤU HÌNH CHUNG
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent

ASSETS_DIR = BASE_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "logo_httm.png"
BANNER_PATH = ASSETS_DIR / "banner.png"

DATA_PATH = BASE_DIR / "data" / "data_motobikes.xlsx"

MODEL_PATHS = {
    "phổ thông": BASE_DIR / "models" / "model_phothong.pkl",
    "trung cấp": BASE_DIR / "models" / "model_trungcap.pkl",
    "cao cấp": BASE_DIR / "models" / "model_caocap.pkl",
}

MODEL_METRICS_CANDIDATES = [
    BASE_DIR / "models" / "model_metrics.csv",
    BASE_DIR / "models" / "model_metrics.json",
    BASE_DIR / "models" / "metrics.csv",
    BASE_DIR / "models" / "metrics.json",
]

SAMPLE_BATCH_PATH = BASE_DIR / "data" / "sample_batch_upload.csv"
REFERENCE_YEAR = 2025

BATCH_REQUIRED_COLUMNS = [
    "thuong_hieu",
    "dong_xe",
    "loai_xe",
    "nam_dang_ky",
    "gia_rao",
]

BATCH_OPTIONAL_COLUMNS = [
    "so_km",
    "dung_tich_xe",
    "xuat_xu",
    "quan",
    "tieu_de",
    "mo_ta",
]

st.set_page_config(
    page_title="Dự đoán giá & Phát hiện bất thường xe máy cũ",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>

/* =======================================================
   FONT TOÀN TRANG
======================================================= */

html,
body,
[data-testid="stAppViewContainer"]{
    font-family: "Inter","Segoe UI","Roboto","Arial",sans-serif;
    background:#F8FAFC;
    color:#243447;
}

/* =======================================================
   VÙNG NỘI DUNG
======================================================= */

.block-container{

    max-width:100%;
    padding-top:0 !important;
    padding-left:.35rem !important;
    padding-right:.35rem !important;
    padding-bottom:1rem !important;

}

main .block-container{padding-top:0 !important;}

/* =======================================================
   SIDEBAR
======================================================= */

[data-testid="stSidebar"]{

    background:#F1F6FD;

    border-right:1px solid #D9E5F3;

    min-width:305px;

    max-width:305px;

}

section[data-testid="stSidebar"] > div:first-child{

    padding:0 !important;

    margin:0 !important;

}

[data-testid="stSidebarContent"]{

    padding-top:0 !important;

    margin-top:0 !important;

}

[data-testid="stSidebarUserContent"]{

    padding-top:0 !important;

    margin-top:0 !important;

}

[data-testid="stSidebar"] .block-container{

    padding:0;

}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4{

    color:#12355B;

    font-weight:750;

}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label{

    color:#475467;

    font-size:15px;

}

[data-testid="stSidebar"] div[role="radiogroup"] label {
    font-size: 1.12rem;
    line-height: 1.5;
    padding: 0.35rem 0;
}

/* =======================================================
   BANNER HTML
======================================================= */

.banner-wrapper {
    position: relative;
    display: block;
    width: 100%;
    margin: 0;
    padding: 0;
    line-height: 0;
    overflow: visible;
}

.banner-wrapper img {
    position: relative;
    display: block;
    width: 100%;
    max-width: 100%;
    height: auto;
    margin: 0;
    padding: 0;
    object-fit: contain;
    object-position: top center;
    border-radius: 14px;
}

/* =======================================================
   HERO
======================================================= */

.hero{

    background:linear-gradient(
        135deg,
        #FFFFFF 0%,
        #F4F8FF 100%
    );

    border-left:6px solid #0B5ED7;

    border:1px solid #D9E5F3;

    border-radius:16px;

    padding:1.3rem 1.6rem;

    margin-top:.5rem;

    margin-bottom:1.2rem;

    box-shadow:0 4px 12px rgba(0,0,0,.05);

}

.hero h1{

    color:#12355B;

    font-size:2rem;

    font-weight:800;

    margin:0;

}

.hero p{

    color:#475467;

    margin-top:0.5rem;

    font-size:1.05rem;

}

/* =======================================================
   CARD
======================================================= */

.info-card,
.result-card{

    background:#FFFFFF;

    border:1px solid #D9E5F3;

    border-radius:14px;

    padding:1rem;

    box-shadow:0 2px 8px rgba(0,0,0,.04);

}

/* =======================================================
   METRIC
======================================================= */

[data-testid="stMetric"]{

    background:#FFFFFF;

    border:1px solid #D9E5F3;

    border-radius:12px;

    padding:12px;

    box-shadow:0 2px 8px rgba(0,0,0,.04);

}

[data-testid="stMetricLabel"] {
    color: #475467;
    font-size: 1rem;
}

[data-testid="stMetricValue"] {
    color: #0B5ED7;
    font-size: 2rem;
    font-weight: 750;
}

[data-testid="stWidgetLabel"] p {
    font-size: 1.02rem;
    font-weight: 600;
    color: #334155;
}

[data-baseweb="input"] input,
[data-baseweb="select"] input,
[data-baseweb="textarea"] textarea {
    font-size: 1rem;
}

/* =======================================================
   BUTTON - st.button()
======================================================= */

/* Nút thông thường */

.stButton > button{

    background:#D32F2F !important;

    color:#FFFFFF !important;

    border:none !important;

    border-radius:10px;

    font-size:1.05rem;

    font-weight:700;

    transition:all .25s ease;

}

/* Chữ và icon */

.stButton > button p,
.stButton > button span,
.stButton > button div{

    color:#FFFFFF !important;

}

.stButton > button svg{

    fill:#FFFFFF !important;

}

/* Hover */

.stButton > button:hover{

    background:#B71C1C !important;

}

/* Active */

.stButton > button:active{

    background:#8E0000 !important;

}


/* =======================================================
   FORM SUBMIT BUTTON - st.form_submit_button()
======================================================= */

button[data-testid="stBaseButton-primaryFormSubmit"]{

    background:#D32F2F !important;

    color:#FFFFFF !important;

    border:none !important;

    border-radius:10px;

    font-size:1.05rem;

    font-weight:700;

    transition:all .25s ease;

}

/* Chữ */

button[data-testid="stBaseButton-primaryFormSubmit"]
div[data-testid="stMarkdownContainer"] p{

    color:#FFFFFF !important;

    font-weight:700 !important;

}

/* Span */

button[data-testid="stBaseButton-primaryFormSubmit"] span{

    color:#FFFFFF !important;

}

/* Icon */

button[data-testid="stBaseButton-primaryFormSubmit"] svg{

    fill:#FFFFFF !important;

}

/* Hover */

button[data-testid="stBaseButton-primaryFormSubmit"]:hover{

    background:#B71C1C !important;

}

button[data-testid="stBaseButton-primaryFormSubmit"]:hover
div[data-testid="stMarkdownContainer"] p{

    color:#FFFFFF !important;

}

/* Active */

button[data-testid="stBaseButton-primaryFormSubmit"]:active{

    background:#8E0000 !important;

}


/* =======================================================
   DOWNLOAD BUTTON - st.download_button()
======================================================= */

.stDownloadButton > button{

    background:#0B5ED7 !important;

    color:#FFFFFF !important;

    border:none !important;

    border-radius:10px;

    font-size:1.02rem;

    font-weight:650;

    transition:all .25s ease;

}

.stDownloadButton > button p,
.stDownloadButton > button span,
.stDownloadButton > button div{

    color:#FFFFFF !important;

}

.stDownloadButton > button svg{

    fill:#FFFFFF !important;

}

.stDownloadButton > button:hover{

    background:#0849A5 !important;

}

.stDownloadButton > button:active{

    background:#063B83 !important;

}

/* =======================================================
   INPUT
======================================================= */

[data-baseweb="input"]>div,
[data-baseweb="select"]>div{

    border-radius:8px;

    border:1px solid #CBD5E1;

}

/* =======================================================
   DATAFRAME
======================================================= */

[data-testid="stDataFrame"]{

    border-radius:12px;

    border:1px solid #D9E5F3;

}

/* =======================================================
   TAB
======================================================= */

button[data-baseweb="tab"] {
    font-size: 1.08rem;
    font-weight: 650;
    padding-left: 1rem;
    padding-right: 1rem;
    min-height: 3.2rem;
    color: #475467;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #E53935;
    font-weight: 750;
}

/* Chữ bên trong tab */
button[data-baseweb="tab"] p {
    font-size: 1.08rem;
}

/* =======================================================
   TITLE
======================================================= */

h1{

    color:#12355B;

    font-weight:800;

}

h2{

    color:#12355B;

}

h3{

    color:#12355B;

}

/* =======================================================
   NỘI DUNG VĂN BẢN
======================================================= */

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {

    font-size:1.08rem;

    line-height:1.75;

    color:#334155;

}

/* Không áp dụng màu cho Markdown nằm trong nút */

button[data-testid="stBaseButton-primaryFormSubmit"]
div[data-testid="stMarkdownContainer"] p{

    color:#FFFFFF !important;

}

/* Tiêu đề cấp 3 như Bối cảnh kinh doanh, Đặc trưng chính */
[data-testid="stMarkdownContainer"] h3 {
    font-size: 1.55rem;
    line-height: 1.35;
    margin-top: 1.2rem;
    margin-bottom: 0.65rem;
}

/* Tiêu đề cấp 4 */
[data-testid="stMarkdownContainer"] h4 {
    font-size: 1.25rem;
    line-height: 1.4;
}

/* =======================================================
   FOOTER
======================================================= */

.footer{

    color:#667085;

    text-align:center;

    font-size:14px;

    padding:18px;

}

.small-note{

    color:#667085;

    font-size:14px;

}

*{

    box-sizing:border-box;

}

/* =======================================================
   ẨN HEADER MẶC ĐỊNH CỦA STREAMLIT
======================================================= */

header[data-testid="stHeader"] {
    display: none !important;
} 

div[data-testid="stToolbar"] {
    display: none !important;
}

[data-testid="stDecoration"] {
    display: none !important;
}

[data-testid="stAppViewContainer"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

[data-testid="stAppViewContainer"] > .main {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

section[data-testid="stSidebar"] > div{

    padding-top:0 !important;

    margin-top:0 !important;

}

#MainMenu{

    visibility:hidden;

}

footer{

    visibility:hidden;

}

/* =======================================================
   LOGO SIDEBAR
======================================================= */

.sidebar-logo{

    width:calc(100% + 2rem);

    margin:-4rem -1rem 0.8rem -1rem;

    padding:0;

    background:#FFFFFF;

    border-bottom:1px solid #D9E5F3;

    overflow:hidden;

}

.sidebar-logo img{

    display:block;

    width:99%;

    height:auto;

    margin:0;

}

</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# show_banner()


# =====================================================================
# 2. HÀM TIỆN ÍCH GIAO DIỆN
# =====================================================================
def format_million_vnd(value: float | int | None) -> str:
    """Định dạng số triệu đồng cho giao diện."""
    if value is None or pd.isna(value):
        return "Không xác định"
    return f"{float(value):,.1f} triệu".replace(",", ".")


def format_vnd(value: float | int | None) -> str:
    """Định dạng giá trị VND."""
    if value is None or pd.isna(value):
        return "Không xác định"
    return f"{float(value):,.0f} đ".replace(",", ".")


def normalize_prediction_value(value: float) -> float:
    """
    Chuẩn hóa giá dự đoán về đơn vị triệu đồng để hiển thị.

    Notebook P2 huấn luyện target từ cột Giá_clean nhưng dữ liệu có thể được lưu
    theo VND hoặc triệu đồng tùy phiên bản tiền xử lý. Quy ước:
    - giá trị lớn hơn 100.000 được xem là VND;
    - giá trị nhỏ hơn hoặc bằng 100.000 được xem là triệu đồng.
    """
    value = float(value)
    return value / 1_000_000 if abs(value) > 100_000 else value


def safe_select_index(options: list[Any], preferred: Any, default: int = 0) -> int:
    """Lấy index an toàn cho selectbox."""
    try:
        return options.index(preferred)
    except (ValueError, AttributeError):
        return default


def show_figure(fig: Any) -> None:
    """Hiển thị figure nếu module visualization trả về matplotlib figure."""
    if fig is not None:
        st.pyplot(fig, use_container_width=True)


def load_optional_metrics() -> pd.DataFrame:
    """Đọc bảng metrics nếu project đã có file metrics."""
    for path in MODEL_METRICS_CANDIDATES:
        if not path.exists():
            continue

        try:
            if path.suffix.lower() == ".csv":
                return pd.read_csv(path)

            with path.open("r", encoding="utf-8") as file:
                payload = json.load(file)

            if isinstance(payload, list):
                return pd.DataFrame(payload)

            if isinstance(payload, dict):
                if "metrics" in payload and isinstance(payload["metrics"], list):
                    return pd.DataFrame(payload["metrics"])
                return pd.DataFrame(payload).reset_index(names="Phân khúc")
        except Exception as exc:
            st.warning(f"Không thể đọc file metrics `{path.name}`: {exc}")

    return pd.DataFrame()


def render_page_header(title: str, subtitle: str) -> None:
    """Header dùng chung cho mỗi trang."""
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

import base64


def render_banner() -> None:
    """Hiển thị đầy đủ banner bằng HTML thay cho st.image()."""

    if not BANNER_PATH.exists():
        st.warning(f"Không tìm thấy banner: {BANNER_PATH}")
        return

    banner_base64 = base64.b64encode(
        BANNER_PATH.read_bytes()
    ).decode("utf-8")

    st.markdown(
        f"""
        <div class="banner-wrapper">
            <img
                src="data:image/png;base64,{banner_base64}"
                alt="Dự đoán giá xe máy và phát hiện bất thường"
            />
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_sidebar_logo():
    if not LOGO_PATH.exists():
        return

    logo_base64 = base64.b64encode(
        LOGO_PATH.read_bytes()
    ).decode()

    st.markdown(
        f"""
    <div class="sidebar-logo">
        <img src="data:image/png;base64,{logo_base64}" alt="Logo">
    </div>
    """,
        unsafe_allow_html=True,
    )

@st.cache_data(show_spinner="Đang đọc dữ liệu...")
def get_dataset(path: str) -> pd.DataFrame:
    """Đọc dữ liệu bằng module preprocess."""
    return load_dataset(path)


@st.cache_data(show_spinner="Đang chuẩn bị dữ liệu phân tích...")
def get_eda_dataset(path: str) -> pd.DataFrame:
    """Chuẩn bị các cột EDA như giá, năm đăng ký và quận/huyện."""
    raw = load_dataset(path)
    return prepare_eda_data(raw)


@st.cache_resource(show_spinner="Đang tải các mô hình dự đoán...")
def get_price_models() -> dict[str, Any]:
    """Tải ba pipeline dự đoán theo phân khúc."""
    return load_models(
        phothong_path=str(MODEL_PATHS["phổ thông"]),
        trungcap_path=str(MODEL_PATHS["trung cấp"]),
        caocap_path=str(MODEL_PATHS["cao cấp"]),
    )


def render_vehicle_form(
    options: dict[str, list[Any]],
    prefix: str,
    include_asking_price: bool,
) -> dict[str, Any]:
    """Form nhập thông tin một xe, dùng chung cho dự đoán và kiểm tra bất thường."""
    brand_options = options.get("brands", ["Honda"])
    brand = st.selectbox(
        "Thương hiệu *",
        options=brand_options,
        index=safe_select_index(brand_options, "Honda"),
        key=f"{prefix}_brand",
    )

    model_map = options.get("models_by_brand", {})
    model_options = model_map.get(brand, options.get("models", ["Khác"]))
    model_options = model_options or ["Khác"]

    col1, col2 = st.columns(2)
    with col1:
        model_name = st.selectbox(
            "Dòng xe *",
            options=model_options,
            key=f"{prefix}_model",
        )
    with col2:
        vehicle_types = options.get("vehicle_types", ["Xe số", "Tay ga", "Tay côn/Moto"])
        vehicle_type = st.selectbox(
            "Loại xe *",
            options=vehicle_types,
            key=f"{prefix}_vehicle_type",
        )

    col3, col4 = st.columns(2)
    with col3:
        engine_options = options.get(
            "engine_sizes",
            ["Dưới 50 cc", "50 - 100 cc", "100 - 175 cc", "Trên 175 cc", "Không biết rõ"],
        )
        engine_size = st.selectbox(
            "Dung tích xe *",
            options=engine_options,
            index=safe_select_index(engine_options, "100 - 175 cc"),
            key=f"{prefix}_engine",
        )
    with col4:
        origins = options.get("origins", ["Việt Nam", "Nhật Bản", "Thái Lan", "Đang cập nhật"])
        origin = st.selectbox(
            "Xuất xứ *",
            options=origins,
            index=safe_select_index(origins, "Việt Nam"),
            key=f"{prefix}_origin",
        )

    current_year = dt.date.today().year
    col5, col6 = st.columns(2)
    with col5:
        before_1980 = st.checkbox(
            "Xe đăng ký trước năm 1980",
            value=False,
            key=f"{prefix}_before_1980",
        )
        registration_year = st.number_input(
            "Năm đăng ký *",
            min_value=1980,
            max_value=current_year,
            value=min(2018, current_year),
            step=1,
            disabled=before_1980,
            key=f"{prefix}_year",
        )
    with col6:
        
        st.info(
                "Các mô hình hiện tại được huấn luyện trên những bản ghi có số km hợp lệ. "
                "Nếu không rõ chính xác, vui lòng nhập một giá trị ước tính cho Số km đã đi vào ô sau đây để thực hiện dự đoán."
                )
        
        kilometers = st.number_input(
            "Số km đã đi *",
            min_value=0,
            max_value=999_999,
            value=26_000,
            step=1_000,
            key=f"{prefix}_km",
        )
        
    districts = options.get("districts", ["Không rõ"])
    district = st.selectbox(
        "Quận/Huyện giao dịch *",
        options=districts,
        index=safe_select_index(districts, "Thành phố Thủ Đức"),
        key=f"{prefix}_district",
    )

    st.markdown("#### Thông tin tin đăng")
    title = st.text_input(
        "Tiêu đề",
        placeholder="Ví dụ: Honda Vision 2018 chính chủ, xe đẹp, máy zin",
        key=f"{prefix}_title",
    )
    description = st.text_area(
        "Mô tả chi tiết",
        placeholder=(
            "Ví dụ: Xe chính chủ, BSTP, máy zin nguyên bản, không ngập nước, "
            "không đâm đụng, giấy tờ đầy đủ, còn thương lượng..."
        ),
        height=130,
        key=f"{prefix}_description",
    )

    asking_price = None
    if include_asking_price:
        asking_price = st.number_input(
            "Giá rao bán hiện tại (triệu đồng) *",
            min_value=0.5,
            max_value=2_000.0,
            value=30.0,
            step=0.5,
            key=f"{prefix}_asking_price",
        )

    return {
        "Thương hiệu": brand,
        "Dòng xe": model_name,
        "Loại xe": vehicle_type,
        "Dung tích xe": engine_size,
        "Xuất xứ": origin,
        "Năm đăng ký": "Trước năm 1980" if before_1980 else int(registration_year),
        "Số Km đã đi": int(kilometers),
        "Quan": district,
        "Địa chỉ": district,
        "Tiêu đề": title.strip(),
        "Mô tả chi tiết": description.strip(),
        "Giá rao": asking_price,
        "reference_year": REFERENCE_YEAR,
    }


def render_prediction_result(result: dict[str, Any]) -> None:
    """Hiển thị kết quả dự đoán giá một xe."""
    predicted = normalize_prediction_value(result["predicted_price"])
    low = normalize_prediction_value(result.get("lower_bound", predicted * 0.9))
    high = normalize_prediction_value(result.get("upper_bound", predicted * 1.1))
    segment = result.get("segment", "Không xác định")
    model_name = result.get("model_name", "XGBoost pipeline")

    st.success("Mô hình đã dự đoán thành công.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Giá đề xuất", format_million_vnd(predicted))
    col2.metric("Cận tham khảo thấp", format_million_vnd(max(low, 0)))
    col3.metric("Cận tham khảo cao", format_million_vnd(max(high, 0)))

    st.markdown(
        f"""
        <div class="result-card">
            <b>Phân khúc được nhận diện:</b> {segment}<br>
            <b>Mô hình dự đoán được sử dụng:</b> {model_name}<br>
            <span class="small-note">
                Kết quả được suy ra từ cột "Giá" trong dữ liệu Chợ Tốt; đây là giá tham khảo,
                không thay thế kiểm tra thực tế về máy móc, giấy tờ và lịch sử sử dụng.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_anomaly_result(result: dict[str, Any], asking_price_million: float) -> None:
    """Hiển thị kết quả phân tích bất thường."""
    predicted = normalize_prediction_value(result["predicted_price"])
    residual = float(asking_price_million) - predicted
    deviation_pct = residual / predicted * 100 if predicted else 0.0

    score = float(result.get("anomaly_score", 0.0))
    threshold = float(result.get("threshold", 95.0))
    label = str(result.get("label", "Bình thường"))
    is_anomaly = bool(result.get("is_anomaly", False))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Giá rao", format_million_vnd(asking_price_million))
    col2.metric("Giá đề xuất", format_million_vnd(predicted))
    col3.metric(
        "Chênh lệch",
        format_million_vnd(abs(residual)),
        delta=f"{deviation_pct:+.1f}%",
        delta_color="inverse",
    )
    col4.metric("Điểm bất thường", f"{score:.1f}/100")

    if not is_anomaly:
        st.success(f"✅ Kết luận: **{label}**")
    elif residual < 0:
        # st.warning(f"⚠️ Kết luận: **{label} — mức giá có dấu hiệu quá rẻ**")
        st.warning(f"⚠️ Kết luận: Mức giá có dấu hiệu **{label}**")
    else:
        st.error(f"🚨 Kết luận: Mức giá có dấu hiệu **{label}**")

    st.caption(
            f"Ngưỡng cảnh báo: {threshold:.1f}/100, được xác định từ chính dữ liệu thị trường. "
            "Sau khi hệ thống tính điểm bất thường cho toàn bộ tin đăng, phân vị 95% của các điểm được chọn làm ngưỡng cảnh báo. "
            "Vì vậy chỉ khoảng 5% tin đăng có điểm bất thường cao nhất mới được gắn cờ để người dùng xem xét kỹ hơn."
            )

    if score < threshold:
        st.success(
            f"Điểm bất thường ({score:.1f}) thấp hơn ngưỡng cảnh báo ({threshold:.1f}), nên tin đăng được đánh giá là Bình thường."
        )
    else:
        st.error(
            f"Điểm bất thường ({score:.1f}) vượt ngưỡng cảnh báo ({threshold:.1f}), nên tin đăng được gắn cờ bất thường."
        )
  
    show_figure(
        plot_prediction_comparison(
            predicted_price=predicted,
            asking_price=asking_price_million,
        )
    )

    components = result.get("components")
    
    if isinstance(components, dict) and components:
        show_figure(
            plot_anomaly_score_components(
                components,
                threshold=threshold,
            )
        )

        from src.anomaly_detector import (
            WEIGHT_RESID,
            WEIGHT_MINMAX,
            WEIGHT_RANGE,
            WEIGHT_ISO,
        )

        score_df = pd.DataFrame(
            {
                "Tín hiệu": [
                    "Residual-Z",
                    "P1/P99",
                    "P10/P90",
                    "Isolation Forest",
                ],
                "Trọng số": [
                    WEIGHT_RESID,
                    WEIGHT_MINMAX,
                    WEIGHT_RANGE,
                    WEIGHT_ISO,
                ],
                "Điểm": [
                    float(components.get("Residual-z", 0.0)),
                    float(components.get("P1/P99", 0.0)),
                    float(components.get("P10/P90", 0.0)),
                    float(components.get("Isolation Forest", 0.0)),
                ],
            }
        )

        score_df["Đóng góp"] = (
            score_df["Trọng số"] * score_df["Điểm"]
        )

        st.subheader("Cách tính điểm bất thường")

        st.dataframe(
            score_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Trọng số": st.column_config.NumberColumn(
                    "Trọng số",
                    format="%.2f",
                ),
                "Điểm": st.column_config.NumberColumn(
                    "Điểm",
                    format="%.1f",
                ),
                "Đóng góp": st.column_config.NumberColumn(
                    "Đóng góp vào điểm tổng",
                    format="%.1f",
                ),
            },
        )

        st.success(
            f"Điểm bất thường = {score:.1f}/100"
        )

# =====================================================================
# 3. KIỂM TRA TÀI NGUYÊN VÀ LOAD DỮ LIỆU
# =====================================================================
missing_resources: list[str] = []

if not DATA_PATH.exists():
    missing_resources.append(str(DATA_PATH.relative_to(BASE_DIR)))

for model_path in MODEL_PATHS.values():
    if not model_path.exists():
        missing_resources.append(str(model_path.relative_to(BASE_DIR)))

if missing_resources:
    st.error(
        "Project đang thiếu các tập tin bắt buộc:\n\n- "
        + "\n- ".join(missing_resources)
    )
    st.info(
        "Hãy kiểm tra lại cấu trúc `data/` và `models/`, sau đó tải lại ứng dụng."
    )
    st.stop()

try:
    df_raw = get_dataset(str(DATA_PATH))
    df_eda = get_eda_dataset(str(DATA_PATH))
    models = get_price_models()
except Exception as exc:
    st.error("Không thể khởi tạo ứng dụng.")
    st.exception(exc)
    st.stop()

try:
    form_options = get_form_options(df_raw)
except Exception:
    # Fallback tối thiểu để giao diện vẫn render khi module chưa hoàn chỉnh.
    brands = sorted(df_raw["Thương hiệu"].dropna().astype(str).unique().tolist())
    models_by_brand = {
        brand: sorted(
            df_raw.loc[df_raw["Thương hiệu"].astype(str) == brand, "Dòng xe"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        for brand in brands
    }
    form_options = {
        "brands": brands,
        "models_by_brand": models_by_brand,
        "models": sorted(df_raw["Dòng xe"].dropna().astype(str).unique().tolist()),
        "vehicle_types": sorted(df_raw["Loại xe"].dropna().astype(str).unique().tolist()),
        "engine_sizes": sorted(df_raw["Dung tích xe"].dropna().astype(str).unique().tolist()),
        "origins": sorted(df_raw["Xuất xứ"].dropna().astype(str).unique().tolist()),
        "districts": sorted(
            df_raw["Địa chỉ"]
            .dropna()
            .apply(extract_district)
            .astype(str)
            .unique()
            .tolist()
        ),
    }


# =====================================================================
# 4. SIDEBAR
# =====================================================================
with st.sidebar:
    
    render_sidebar_logo()
        
    st.title("🏍️ MOTORBIKE ML")
    st.caption("Dự đoán giá và phát hiện bất thường")

    menu = st.radio(
        "MENU",
        [
            "Thách thức doanh nghiệp",
            "Đánh giá & Báo cáo",
            "Dự báo / Phân tích / Khuyến nghị",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("#### Tài nguyên")
    st.write(f"**Dữ liệu:** `{DATA_PATH.name}`")
    st.write(f"**Số tin đăng:** {len(df_raw):,}")
    st.write(f"**Số thuộc tính:** {df_raw.shape[1]}")
    st.write("-------------------------------------")
    st.write(f"**Đồ Án Tốt Nghiệp - DL07 - K314**")
    st.write(f"**Nhóm thực hiện:** Phan Kim Thanh_Nguyễn Quang Lợi")
    st.write("")
    
    
    st.divider()
    # st.caption(
        # "Target của mô hình là asking price trong tin đăng, "
        # "không phải giá giao dịch cuối cùng."
    # )


# =====================================================================
# 5. BUSINESS PROBLEM
# =====================================================================
if menu == "Thách thức doanh nghiệp":
    
    render_banner()
    
    render_page_header(
        "🏍️ Dự đoán giá và phát hiện bất thường cho các tin đăng bán xe máy cũ",
        "Ứng dụng Mô Hình Học Máy (Machine Learning) cho dữ liệu tin đăng tại TP.HCM trên Chợ Tốt.",
    )

    st.markdown(
        """
        ### Bối cảnh kinh doanh

        Người bán thường khó xác định mức giá phù hợp cho xe máy cũ vì giá phụ thuộc đồng thời
        vào thương hiệu, dòng xe, tuổi xe, số km đã đi, dung tích, xuất xứ, khu vực và thông tin
        mô tả. Người mua cũng cần nhận biết những tin có mức giá lệch đáng kể so với các xe tương đồng.

        Cột **Giá** trong dữ liệu là mức giá người bán mong muốn. Vì vậy, mô hình không khẳng định
        đây là “giá thị trường tuyệt đối”, mà học từ mặt bằng tin đăng để đưa ra một **giá đề xuất**
        và dùng phần sai lệch giữa giá rao với giá đề xuất để hỗ trợ phát hiện bất thường.
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="info-card">
                <h4>🎯 Bài toán 1</h4>
                <b>Dự đoán giá đề xuất</b>
                <p>Ước lượng mức giá phù hợp từ thông tin của một xe máy cũ.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="info-card">
                <h4>🚨 Bài toán 2</h4>
                <b>Phát hiện bất thường giá</b>
                <p>Đánh dấu tin có dấu hiệu quá rẻ hoặc quá đắt so với nhóm tham chiếu.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="info-card">
                <h4>💡 Giá trị sử dụng</h4>
                <b>Phân tích và khuyến nghị</b>
                <p>Hỗ trợ người mua, người bán và bộ phận kiểm duyệt tin đăng.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Quy trình Data Science")
    st.code(
        """
Đọc dữ liệu
    ↓
Phân tích khám phá dữ liệu (Exploratory Data Analysis - EDA)
    ↓
Làm sạch giá, năm đăng ký, số km và địa chỉ
    ↓
Xây dựng đặc trưng (Feature Engineering)
    ↓
Huấn luyện 3 mô hình theo phân khúc Phổ thông/ Trung cấp/ Cao cấp 
    ↓
Dự đoán giá đề xuất
    ↓
Tính 4 tín hiệu (Residual-z, Vi phạm P1/P99, Ngoài P10/P90 và Isolation Forest) để phát hiện bất thường
    ↓
Điểm tổng hợp và khuyến nghị
        """.strip(),
        language="text",
    )

    st.markdown("### Đặc trưng chính")
    feature_col1, feature_col2 = st.columns(2)
    with feature_col1:
        st.markdown(
            """
            - Thương hiệu, dòng xe, loại xe, dung tích và xuất xứ.
            - Tuổi xe, số km, km trung bình mỗi năm.
            - Quận/Huyện giao dịch.
            - Phân khúc xe: Hệ thống sử dụng kiến thức thị trường xe máy Việt Nam để phân khúc xe được xây dựng từ "Thương Hiệu" và "Dòng Xe" để phân khúc phổ thông, trung cấp và cao cấp.
            """
        )
    with feature_col2:
        st.markdown(
            """
            - Chính chủ, BSTP, sang tên, còn bảo hành.
            - Xe zin, nhập khẩu, ít sử dụng, không ngập.
            - Độ dài và nội dung mô tả tin đăng.
            - Trong quá trình huấn luyện, giá xe được chuyển sang thang đo logarit (log1p) để giảm sự chênh lệch giữa các mức giá, giúp mô hình học ổn định và dự đoán chính xác hơn. Sau khi dự đoán, kết quả được chuyển đổi trở lại thành giá tiền thực tế để hiển thị cho người dùng.
            """
        )

    st.info(
        "Ứng dụng mang tính hỗ trợ ra quyết định. Trước khi mua hoặc bán xe, "
        "cần kiểm tra trực tiếp tình trạng máy, khung sườn, giấy tờ và lịch sử sử dụng."
    )


# =====================================================================
# 6. EVALUATION & REPORT
# =====================================================================
elif menu == "Đánh giá & Báo cáo":
    
    render_banner()
    
    render_page_header(
        "📊 Đánh giá & Báo cáo",
        "Tổng quan dữ liệu, phân tích khám phá và kết quả đánh giá mô hình.",
    )

    quality = get_data_quality_summary(df_raw)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Số dòng", f"{quality.get('rows', len(df_raw)):,}")
    col2.metric("Số cột", f"{quality.get('columns', df_raw.shape[1]):,}")
    col3.metric("Dòng trùng", f"{quality.get('duplicates', 0):,}")
    col4.metric("Ô dữ liệu thiếu", f"{quality.get('missing_cells', int(df_raw.isna().sum().sum())):,}")

    overview_tab, eda_tab, model_tab = st.tabs(
        ["📋 Tổng quan dữ liệu", "📈 EDA", "🤖 Báo cáo mô hình"]
    )

    with overview_tab:
        st.subheader("Xem trước dữ liệu")
        preview_rows = st.slider(
            "Số dòng hiển thị",
            min_value=5,
            max_value=min(100, len(df_raw)),
            value=min(10, len(df_raw)),
            step=5,
        )
        st.dataframe(df_raw.head(preview_rows), use_container_width=True)
        
        st.markdown("""       
                        - Các tin đăng xe máy cũ có mức giá rao bán khá đa dạng, phản ánh sự khác biệt về thương hiệu, dòng xe, năm sản xuất, số km đã đi và tình trạng xe.
                        - Dữ liệu bao gồm nhiều thương hiệu (Honda, Yamaha, Piaggio, SYM...), nhiều đời xe (2003–2024) và mức sử dụng khác nhau, giúp mô hình học được đặc điểm của nhiều phân khúc xe máy cũ trên thị trường.
                        """)
    
        st.subheader("Dữ liệu thiếu")
        missing_table = pd.DataFrame(
            {
                "Số dòng thiếu": df_raw.isna().sum(),
                "Tỷ lệ thiếu (%)": (df_raw.isna().mean() * 100).round(2),
            }
        ).sort_values("Số dòng thiếu", ascending=False)

        st.dataframe(missing_table, use_container_width=True)
        
        st.markdown("""       
                                - Dữ liệu có mức độ đầy đủ cao, tỷ lệ thiếu rất thấp (<3%).
                                - Chỉ hai trường Khoảng giá min và Khoảng giá max có tỷ lệ thiếu khoảng 2,8%, các thuộc tính còn lại gần như đầy đủ, đảm bảo độ tin cậy cho quá trình huấn luyện mô hình.
                                """)

        st.subheader("Thống kê mô tả")
        st.dataframe(
            df_eda.select_dtypes(include=np.number).describe().T,
            use_container_width=True,
        )
        
        st.markdown("""       
                                - Dữ liệu gồm 7.208 tin đăng, tập trung chủ yếu ở phân khúc xe với giá trung vị 16,5 triệu đồng, tuổi xe khoảng 10 năm và quãng đường đã đi khoảng 28.000 km.
                                - Đồng thời, dữ liệu xuất hiện một số giá trị ngoại lệ về giá và số km, do đó đã được xử lý trong giai đoạn tiền xử lý để nâng cao chất lượng mô hình.
                                """)    

    with eda_tab:
        eda_choice = st.selectbox(
            "Chọn biểu đồ",
            [
                "Phân bố thương hiệu",
                "Top Quận/Huyện giao dịch",
                "Phân phối giá",
                "Giá theo thương hiệu",
                "Giá theo loại xe",
                "Năm đăng ký và giá",
                "Ma trận tương quan",
                "Phân bố phân khúc xe",
            ],
        )

        if eda_choice == "Phân bố thương hiệu":
            show_figure(plot_brand_distribution(df_eda))
            st.markdown("""
                ### Nhận xét

                - Honda có số lượng tin đăng cao nhất (4.374 tin), chiếm ưu thế trong tập dữ liệu.
                - Yamaha đứng thứ hai với 1.411 tin đăng.
                - Piaggio, Suzuki và SYM có số lượng tin đăng ở mức trung bình.
                - Các thương hiệu còn lại xuất hiện với tần suất thấp.
                - Dữ liệu phản ánh đúng thực trạng thị trường xe máy Việt Nam, nơi Honda và Yamaha là hai thương hiệu phổ biến nhất.
                """)
        elif eda_choice == "Top Quận/Huyện giao dịch":
            show_figure(plot_district_distribution(df_eda))
        elif eda_choice == "Phân phối giá":
            show_figure(plot_price_distribution(df_eda))
            st.markdown("""
                ### Nhận xét

                - Phần lớn xe máy cũ có giá dưới 30 triệu đồng và tập trung nhiều nhất trong khoảng 10–20 triệu đồng.
                - Giá trung vị khoảng 16,5 triệu đồng, cho thấy 50% số tin đăng có giá thấp hơn hoặc bằng mức này.
                - Phân phối giá lệch phải, với một số ít xe có giá rất cao.
                - Biểu đồ chỉ hiển thị đến **percentile 99%**, tức là loại bỏ 1% tin đăng có giá cao nhất để giảm ảnh hưởng của các giá trị ngoại lệ và phản ánh rõ hơn xu hướng chung của thị trường.
                """)
        elif eda_choice == "Giá theo thương hiệu":
            show_figure(plot_price_by_brand(df_eda))
            st.markdown("""
                ### Nhận xét

                - Biểu đồ cho thấy mỗi thương hiệu có một khoảng giá đặc trưng khác nhau.
                - Các thương hiệu mô tô cao cấp như Harley Davidson và Ducati có mức giá cao hơn nhiều so với các thương hiệu phổ thông như Honda, Yamaha hay SYM.
                - Điều này khẳng định rằng "thương hiệu" là một trong những yếu tố quan trọng ảnh hưởng đến giá xe máy cũ và cần được đưa vào mô hình dự đoán.
                """)
        elif eda_choice == "Giá theo loại xe":
            show_figure(plot_price_by_vehicle_type(df_eda))
            st.markdown("""
                ### Nhận xét

                - Biểu đồ cho thấy giá xe máy cũ có sự khác biệt rõ rệt giữa các loại xe.
                - Xe số chủ yếu thuộc phân khúc giá thấp, trong khi xe tay ga và xe tay côn/Moto có mức giá cao hơn và biến động lớn hơn.
                - Điều này cho thấy loại xe là một yếu tố quan trọng cần được đưa vào mô hình để nâng cao độ chính xác của dự đoán giá.
                """)
        elif eda_choice == "Năm đăng ký và giá":
            show_figure(plot_year_price_scatter(df_eda))
        elif eda_choice == "Ma trận tương quan":
            show_figure(plot_correlation_heatmap(df_eda))
            
            st.markdown("""
                            ### Nhận xét
            
                            - Tuổi xe và Năm đăng ký có tương quan nghịch hoàn hảo (-1.00), chỉ cần giữ một biến để tránh trùng lặp thông tin.
                            - Số Km đã đi và Km_clean có tương quan dương hoàn hảo (1.00), nên chỉ sử dụng Km_clean trong mô hình.
                            - Giá xe có tương quan tuyến tính rất thấp với các biến số, cho thấy giá chịu ảnh hưởng tổng hợp của nhiều yếu tố.
                            - ID không có ý nghĩa dự báo nên được loại bỏ khi huấn luyện mô hình.
                            - Ngoài các cặp biến trùng thông tin, dữ liệu không có đa cộng tuyến đáng kể, phù hợp để xây dựng mô hình dự đoán.
                            """)
        else:
            show_figure(plot_segment_distribution(df_eda))

        st.caption(
            "Các biểu đồ sử dụng dữ liệu sau khi chuẩn hóa sơ bộ giá, năm đăng ký "
            "và số km để phục vụ EDA."
        )

    with model_tab:
        st.markdown(
            """
            Dự án thử nghiệm nhiều thuật toán gồm Linear Regression, Decision Tree,
            Random Forest, XGBoost, LightGBM, CatBoost và SVR. 
            
            Các mô hình dự đoán được xây dựng và huấn luyện riêng cho từng phân khúc xe trước khi được lưu trong thư mục models/ để phục vụ triển khai hệ thống..
            """
        )

        metrics_df = load_optional_metrics()
        if metrics_df.empty:
            st.warning(
                "Chưa tìm thấy file metrics trong thư mục `models/`. "
                "Khi hoàn thiện bước đánh giá, hãy lưu bảng kết quả thành "
                "`models/model_metrics.csv` để ứng dụng hiển thị tự động."
            )
            st.code(
                "Phân khúc,Model,MAE (triệu),RMSE (triệu),R2\n"
                "phổ thông,XGBoost,...,...,...\n"
                "trung cấp,XGBoost,...,...,...\n"
                "cao cấp,XGBoost,...,...,...",
                language="text",
            )
        else:
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            show_figure(plot_model_metrics(metrics_df))

        st.markdown("#### Mô hình triển khai")
        deployed_df = pd.DataFrame(
            [
                {
                    "Phân khúc": segment,
                    "File": path.name,
                    "Trạng thái": "Sẵn sàng" if path.exists() else "Thiếu file",
                }
                for segment, path in MODEL_PATHS.items()
            ]
        )
        st.dataframe(deployed_df, use_container_width=True, hide_index=True)


# =====================================================================
# 7. NEW PREDICTION / ANALYSIS / RECOMMENDATION
# =====================================================================
else:
    
    render_banner()
    
    render_page_header(
        "🔮 Dự báo / Phân tích / Khuyến nghị",
        "Dự đoán một xe, kiểm tra mức giá rao hoặc phân tích hàng loạt từ CSV/XLSX.",
    )

    prediction_tab, anomaly_tab, batch_tab = st.tabs(
        [
            "💰 Dự đoán giá",
            "🚨 Phát hiện bất thường",
            "📂 Kiểm tra hàng loạt",
        ]
    )

    # -----------------------------------------------------------------
    # TAB 1: DỰ ĐOÁN GIÁ
    # -----------------------------------------------------------------
    with prediction_tab:
        st.subheader("Nhập thông tin xe")
        with st.form("single_prediction_form", clear_on_submit=False):
            prediction_values = render_vehicle_form(
                form_options,
                prefix="prediction",
                include_asking_price=False,
            )
            submit_prediction = st.form_submit_button(
                "💰 Dự đoán giá đề xuất",
                use_container_width=True,
                type="primary",
                key="btn_prediction",
            )

        if submit_prediction:
            try:
                input_df = build_input_dataframe(prediction_values, reference_df=df_raw)
                result = predict_price(
                    input_df=input_df,
                    models=models,
                    reference_df=df_raw,
                )

                st.session_state["last_prediction"] = result
                st.session_state["last_prediction_input"] = prediction_values
            except Exception as exc:
                st.error("Không thể thực hiện dự đoán.")
                st.exception(exc)

        if "last_prediction" in st.session_state:
            st.divider()
            st.subheader("Kết quả dự đoán")
            render_prediction_result(st.session_state["last_prediction"])

            with st.expander("Xem dữ liệu đầu vào"):
                display_input = pd.DataFrame(
                    [st.session_state["last_prediction_input"]]
                ).drop(columns=["Giá rao", "reference_year"], errors="ignore")
                st.dataframe(display_input, use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------
    # TAB 2: PHÁT HIỆN BẤT THƯỜNG
    # -----------------------------------------------------------------
    with anomaly_tab:
        st.subheader("Nhập thông tin xe và giá rao")
        with st.form("single_anomaly_form", clear_on_submit=False):
            anomaly_values = render_vehicle_form(
                form_options,
                prefix="anomaly",
                include_asking_price=True,
            )
            submit_anomaly = st.form_submit_button(
                "🔎 Kiểm tra mức giá",
                use_container_width=True,
                type="primary",
                key="btn_anomaly",
            )

        if submit_anomaly:
            try:
                asking_price = float(anomaly_values["Giá rao"])
                input_df = build_input_dataframe(anomaly_values, reference_df=df_raw)

                prediction_result = predict_price(
                    input_df=input_df,
                    models=models,
                    reference_df=df_raw,
                )

                anomaly_result = analyze_price_anomaly(
                    input_df=input_df,
                    asking_price=asking_price,
                    prediction_result=prediction_result,
                    reference_df=df_eda,
                )

                recommendation = generate_recommendation(
                    predicted_price=normalize_prediction_value(
                        prediction_result["predicted_price"]
                    ),
                    asking_price=asking_price,
                    anomaly_result=anomaly_result,
                    vehicle_info=anomaly_values,
                )

                st.session_state["last_anomaly"] = anomaly_result
                st.session_state["last_anomaly_recommendation"] = recommendation
                st.session_state["last_anomaly_price"] = asking_price
            except Exception as exc:
                st.error("Không thể phân tích mức giá.")
                st.exception(exc)

        if "last_anomaly" in st.session_state:
            st.divider()
            st.subheader("Kết quả phân tích")
            render_anomaly_result(
                st.session_state["last_anomaly"],
                st.session_state["last_anomaly_price"],
            )

            recommendation = st.session_state["last_anomaly_recommendation"]

            st.subheader("Khuyến nghị")

            if isinstance(recommendation, dict):
                level = recommendation.get("level", "info")
                message = recommendation.get("message", "")

                if level == "success":
                    st.success(message)
                elif level == "warning":
                    st.warning(message)
                elif level == "error":
                    st.error(message)
                else:
                    st.info(message)
            else:
                st.info(str(recommendation))

    # -----------------------------------------------------------------
    # TAB 3: KIỂM TRA HÀNG LOẠT
    # -----------------------------------------------------------------
    with batch_tab:
        st.subheader("Tải lên danh sách xe")
        st.markdown(
            """
            File hàng loạt phải có tối thiểu các cột:

            `thuong_hieu`, `dong_xe`, `loai_xe`, `nam_dang_ky`, `gia_rao`

            Các cột tùy chọn:

            `so_km`, `dung_tich_xe`, `xuat_xu`, `quan`, `tieu_de`, `mo_ta`
            """
        )

        if SAMPLE_BATCH_PATH.exists():
            st.download_button(
                "⬇️ Tải file CSV mẫu",
                data=SAMPLE_BATCH_PATH.read_bytes(),
                file_name=SAMPLE_BATCH_PATH.name,
                mime="text/csv",
            )

        uploaded_file = st.file_uploader(
            "Chọn file CSV hoặc XLSX",
            type=["csv", "xlsx"],
            accept_multiple_files=False,
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.lower().endswith(".csv"):
                    batch_df = pd.read_csv(uploaded_file)
                else:
                    batch_df = pd.read_excel(uploaded_file)

                st.markdown("#### Dữ liệu đã tải lên")
                st.dataframe(batch_df.head(30), use_container_width=True)

                validation = validate_batch_columns(
                    batch_df,
                    required_columns=BATCH_REQUIRED_COLUMNS,
                    optional_columns=BATCH_OPTIONAL_COLUMNS,
                )

                if isinstance(validation, tuple):
                    is_valid, validation_message = validation
                elif isinstance(validation, dict):
                    is_valid = bool(validation.get("is_valid", False))
                    validation_message = validation.get("message", "")
                else:
                    is_valid = bool(validation)
                    validation_message = ""

                if not is_valid:
                    st.error(
                        validation_message
                        or "File chưa có đủ các cột bắt buộc."
                    )
                else:
                    st.success("Cấu trúc file hợp lệ.")

                    run_batch = st.button(
                        "⚙️ Chạy dự đoán và phân tích hàng loạt",
                        use_container_width=True,
                        type="primary",
                    )

                    if run_batch:
                        with st.spinner("Đang xử lý danh sách xe..."):
                            batch_prediction_df = predict_batch(
                                batch_df=batch_df,
                                models=models,
                                reference_df=df_raw,
                            )

                            batch_result_df = analyze_batch_anomalies(
                                prediction_df=batch_prediction_df,
                                reference_df=df_eda,
                            )

                            batch_result_df = generate_batch_recommendations(
                                batch_result_df
                            )

                        st.session_state["batch_result_df"] = batch_result_df
            except Exception as exc:
                st.error("Không thể đọc hoặc xử lý file đã tải lên.")
                st.exception(exc)

        if "batch_result_df" in st.session_state:
            result_df = st.session_state["batch_result_df"]

            st.divider()
            st.subheader("Kết quả hàng loạt")

            anomaly_col = (
                result_df["is_anomaly"]
                if "is_anomaly" in result_df.columns
                else pd.Series(False, index=result_df.index)
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Tổng số xe", f"{len(result_df):,}")
            c2.metric("Tin bất thường", f"{int(anomaly_col.sum()):,}")
            c3.metric(
                "Tỷ lệ bất thường",
                f"{anomaly_col.mean() * 100:.1f}%" if len(result_df) else "0,0%",
            )
            
            from datetime import datetime
            
            # 1. KIỂM TRA CÁC CỘT CẦN THIẾT
            required_columns = [
                "Thương hiệu",
                "Dòng xe",
                "Loại xe",
                "Dung tích xe",
                "Xuất xứ",
                "Giá_clean",
                "gia_du_doan",
                "anomaly_score",
                "flag_final",
                "flag_minmax",
            ]
            
            missing_columns = [
                column
                for column in required_columns
                if column not in result_df.columns
            ]
            
            if missing_columns:
                st.error(
                    "Thiếu các cột cần thiết trong kết quả: "
                    + ", ".join(missing_columns)
                )

                with st.expander("Xem danh sách cột hiện có"):
                    st.write(result_df.columns.tolist())

                st.stop()
            
            # 2. TẠO DATAFRAME DASHBOARD
            display_df = result_df[
                [
                    "Thương hiệu",
                    "Dòng xe",
                    "Loại xe",
                    "Dung tích xe",
                    "Xuất xứ",
                    "Giá_clean",
                    "gia_du_doan",
                    "anomaly_score",
                    "flag_final",
                    "flag_minmax",
                ]
            ].copy()
            
            # Đánh số thứ tự
            display_df.insert(
                0,
                "STT",
                range(1, len(display_df) + 1),
            )
            
            # 3. CHUẨN HÓA DỮ LIỆU SỐ
            display_df["Giá_clean"] = pd.to_numeric(
                display_df["Giá_clean"],
                errors="coerce",
            )
            
            display_df["gia_du_doan"] = pd.to_numeric(
                display_df["gia_du_doan"],
                errors="coerce",
            )
            
            display_df["anomaly_score"] = pd.to_numeric(
                display_df["anomaly_score"],
                errors="coerce",
            )
            
            # 4. TÍNH CHÊNH LỆCH
            # Giá rao - Giá mô hình đề xuất
            
            display_df["Chênh lệch"] = (
                display_df["Giá_clean"]
                - display_df["gia_du_doan"]
            )
            
            # 5. ĐỔI TÊN CỘT HIỂN THỊ
            display_df.rename(
                columns={
                    "Giá_clean": "Giá rao",
                    "gia_du_doan": "Giá mô hình đề xuất",
                    "anomaly_score": "Điểm bất thường",
                    "flag_final": "Kết luận",
                    "flag_minmax": "Tín hiệu giá",
                },
                inplace=True,
            )
            
            # 6. ĐƯA CÁC CỘT VỀ ĐÚNG THỨ TỰ
            display_df = display_df[
                [
                    "STT",
                    "Thương hiệu",
                    "Dòng xe",
                    "Loại xe",
                    "Dung tích xe",
                    "Xuất xứ",
                    "Giá rao",
                    "Giá mô hình đề xuất",
                    "Chênh lệch",
                    "Điểm bất thường",
                    "Kết luận",
                    "Tín hiệu giá",
                ]
            ]
            
            # 7. CHUẨN HÓA KẾT LUẬN
            def convert_result(value):
                value = str(value).strip()

                if value == "Quá rẻ":
                    return "🔴 Quá rẻ"

                if value == "Quá đắt":
                    return "🟠 Quá đắt"

                if value == "Bất thường":
                    return "🔴 Bất thường"

                return "🟢 Bình thường"
            
            display_df["Kết luận"] = (
                display_df["Kết luận"]
                .apply(convert_result)
            )
            
            # 8. TẠO KHUYẾN NGHỊ
            # Ưu tiên dựa trên tín hiệu giá flag_minmax
            
            def create_recommendation(price_signal, conclusion):
                price_signal = str(price_signal).strip()
                conclusion = str(conclusion).strip()

                if price_signal == "Quá rẻ":
                    return "Nên kiểm tra kỹ"

                if price_signal == "Quá đắt":
                    return "Nên thương lượng"

                if conclusion == "🔴 Bất thường":
                    return "Cần kiểm tra thêm"

                return "Giá hợp lý"
            
            display_df["Khuyến nghị"] = display_df.apply(
                lambda row: create_recommendation(
                    row["Tín hiệu giá"],
                    row["Kết luận"],
                ),
                axis=1,
            )
            
            # Có thể bỏ cột kỹ thuật này khỏi Dashboard
            display_df.drop(
                columns=["Tín hiệu giá"],
                inplace=True,
            )
            
            # 9. LÀM TRÒN ĐIỂM BẤT THƯỜNG
            display_df["Điểm bất thường"] = (
                display_df["Điểm bất thường"]
                .round(1)
            )
            
            # 10. TẠO BẢN HIỂN THỊ CÓ FORMAT GIÁ
            # Dữ liệu Giá_clean và gia_du_doan đang ở đơn vị triệu đồng
            
            dashboard_view = display_df.copy()
            
            def price_to_vnd(value):
                """Đổi giá từ triệu đồng sang VND để hiển thị."""
                if pd.isna(value):
                    return np.nan
                
                value = float(value)
                
                # Nếu dữ liệu đã là VND thì giữ nguyên
                if abs(value) > 100_000:
                    return value
                
                return value * 1_000_000
            
            def format_money(value):
                if pd.isna(value):
                    return "Không xác định"
                
                value_vnd = price_to_vnd(value)

                return f"{value_vnd:,.0f}".replace(",", ".")
            
            def format_difference(value):
                if pd.isna(value):
                    return "Không xác định"

                value_vnd = price_to_vnd(value)
                
                sign = "+" if value_vnd > 0 else ""
                formatted = f"{abs(value_vnd):,.0f}".replace(",", ".")
                
                if value_vnd < 0:
                    return f"-{formatted}"

                return f"{sign}{formatted}"
            
            dashboard_view["Giá rao"] = (
                dashboard_view["Giá rao"]
                .apply(format_money)
            )
            
            dashboard_view["Giá mô hình đề xuất"] = (
                dashboard_view["Giá mô hình đề xuất"]
                .apply(format_money)
            )

            dashboard_view["Chênh lệch"] = (
                dashboard_view["Chênh lệch"]
                .apply(format_difference)
            )
            
            # 11. HIỂN THỊ DASHBOARD
            
            st.dataframe(
                dashboard_view,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "STT": st.column_config.NumberColumn(
                        "STT",
                        width="small",
                    ),
                    "Thương hiệu": st.column_config.TextColumn(
                        "Thương hiệu",
                        width="medium",
                    ),
                    "Dòng xe": st.column_config.TextColumn(
                        "Dòng xe",
                        width="medium",
                    ),
                    "Loại xe": st.column_config.TextColumn(
                        "Loại xe",
                        width="medium",
                    ),
                    "Dung tích xe": st.column_config.TextColumn(
                        "Dung tích xe",
                        width="medium",
                    ),
                    "Xuất xứ": st.column_config.TextColumn(
                        "Xuất xứ",
                        width="medium",
                    ),
                    "Giá rao": st.column_config.TextColumn(
                        "Giá rao (VNĐ)",
                        width="medium",
                    ),
                    "Giá mô hình đề xuất": st.column_config.TextColumn(
                        "Giá mô hình đề xuất (VNĐ)",
                        width="medium",
                    ),
                    "Chênh lệch": st.column_config.TextColumn(
                        "Chênh lệch (VNĐ)",
                        width="medium",
                    ),
                    "Điểm bất thường": st.column_config.ProgressColumn(
                        "Điểm bất thường",
                        min_value=0,
                        max_value=100,
                        format="%.1f",
                    ),
                    "Kết luận": st.column_config.TextColumn(
                        "Kết luận",
                        width="medium",
                    ),
                    "Khuyến nghị": st.column_config.TextColumn(
                        "Khuyến nghị",
                        width="medium",
                    ),
                },
            )

            # 12. XUẤT DASHBOARD CSV
            # File Dashboard giống bảng đang hiển thị
            
            dashboard_csv = (
                dashboard_view
                .to_csv(index=False)
                .encode("utf-8-sig")
            )
            
            # 13. XUẤT TECHNICAL CSV
            # Giữ đầy đủ tất cả cột kỹ thuật
            
            technical_df = result_df.copy()
            
            technical_csv = (
                technical_df
                .to_csv(index=False)
                .encode("utf-8-sig")
            )
            
            # 14. TẠO TÊN FILE THEO THỜI GIAN
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            dashboard_filename = (
                f"Dashboard_Motorbike_{timestamp}.csv"
            )
            
            technical_filename = (
                f"Technical_Motorbike_{timestamp}.csv"
            )
            
            # 15. HAI NÚT TẢI SONG SONG
            download_col1, download_col2 = st.columns(2)
            
            with download_col1:
                st.download_button(
                    label="📥 Tải Dashboard CSV",
                    data=dashboard_csv,
                    file_name=dashboard_filename,
                    mime="text/csv",
                    use_container_width=True,
                    key="download_dashboard_csv",
                )
                
            with download_col2:
                st.download_button(
                    label="📥 Tải Technical CSV",
                    data=technical_csv,
                    file_name=technical_filename,
                    mime="text/csv",
                    use_container_width=True,
                    key="download_technical_csv",
                )   
                
# =====================================================================
# 8. FOOTER
# =====================================================================
st.markdown("---")
st.markdown(
    """
    <div class="footer">
        Data Science Project · Chợ Tốt Motorbike Dataset ·
        Price Prediction · Anomaly Detection · Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
