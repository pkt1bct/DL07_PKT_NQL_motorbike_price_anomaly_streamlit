# -*- coding: utf-8 -*-
"""
feature_engineering.py

Port phần Feature Engineering từ notebook P2 sang module dùng cho Streamlit.

Mục tiêu:
- Giữ nguyên tên feature, công thức, bins, labels và regex của notebook P2.
- Tương thích với app.py:
      build_input_dataframe(prediction_values, reference_df=df_raw)
- Tận dụng các hàm parse/normalize đã có trong src/preprocess.py.
- Với dữ liệu dự đoán một dòng, các thống kê học từ dữ liệu train
  (DongXe_freq và median) được lấy từ reference_df thay vì chính dòng dự đoán.

Các feature model dùng theo notebook P2:
- Numeric:
  TuoiXe, SoKm, Km_per_year, Km_unknown, DongXe_freq,
  pct_ngoai_hinh, co_pct_ngoai_hinh, co_chinh_chu, co_bstp,
  da_sang_ten, con_bao_hanh, khong_dam_dung, do_dai_mota,
  Dung_tich_so, log_SoKm, XuatXu_khong_ro, nhap_khau,
  co_do_choi, mot_doi_chu, cong_chung, uy_quyen, con_moi_dep,
  giay_to_day_du, bao_tranh_chap, thanh_ly_gap,
  cam_ket_khong_ngap, co_abs, it_su_dung, tra_gop,
  xe_zin_nguyen_ban, thuong_luong, xe_grab_dich_vu.

- Categorical:
  Thương hiệu, Loại xe, Dung tích xe, Xuất xứ, Quan,
  PhanKhucXe, SoKm_Group, SoKm_Year, Tuoi_xe_group.

- Target encoding riêng trong pipeline:
  Dòng xe.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import re

import numpy as np
import pandas as pd

from src.preprocess import (
    CURRENT_YEAR,
    UNKNOWN_KM,
    extract_district,
    normalize_text,
    parse_km,
    parse_money_to_million,
    parse_year,
)


# ============================================================================
# CẤU HÌNH GIỐNG NOTEBOOK P2
# ============================================================================

REFERENCE_YEAR_P2 = 2025
UNKNOWN_CATEGORY = "Không rõ"
DEFAULT_ENGINE_SIZE = "Không biết rõ"
DEFAULT_ORIGIN = "Đang cập nhật"

SEGMENT_MAP = {
    "Honda": {
        "phổ thông": [
            "wave", "future", "dream", "cub", "67", "chaly", "blade"
        ],
        "trung cấp": [
            "air blade", "vision", "vario", "lead", "click", "spacy",
            "ps", "dylan", "@", "winner x", "winner", "sonic", "Air Blade"
        ],
        "cao cấp": [
            "sh mode", "sh", "pcx", "cb", "cbr"
        ],
    },
    "Yamaha": {
        "phổ thông": [
            "sirius", "jupiter", "taurus", "yaz", "yb125"
        ],
        "trung cấp": [
            "nouvo", "mio", "janus", "luvias", "nozza", "grande",
            "freego", "cuxi", "acruzo", "exciter"
        ],
        "cao cấp": [
            "nvx", "xsr", "tfx", "nm-x", "nmx", "xmax",
            "pg-1", "fz", "mt", "r"
        ],
    },
    "Suzuki": {
        "phổ thông": [
            "viva", "axelo", "smash", "revo"
        ],
        "trung cấp": [
            "hayate", "impulse", "epicuro", "raider", "satria", "stinger"
        ],
        "cao cấp": [
            "gsx", "gz", "gn", "xbike", "gd", "fx125", "en", "sapphire"
        ],
    },
    "SYM": {
        "phổ thông": [
            "attila", "elegant", "angela", "star", "elite",
            "magic", "ez", "amigo", "bonus", "cello"
        ],
        "trung cấp": [
            "elizabeth", "galaxy", "shark", "wolf", "enjoy",
            "husky", "joyride", "venus", "sanda boss"
        ],
    },
    "Piaggio": {
        "trung cấp": [
            "zip", "liberty"
        ],
        "cao cấp": [
            "sprint", "primavera", "lx", "gts", "medley",
            "beverly", "gt", "et8", "fly", "vespa"
        ],
        "siêu cao cấp": [
            "vespa 946"
        ],
    },
    "Kymco": {
        "trung cấp": [
            "candy", "k-pipe", "jockey", "people", "like"
        ],
    },
}

MIXED_MOTO_BRANDS = ["Kawasaki", "KTM", "GPX", "BMW"]
LUXURY_MOTO_BRANDS = ["Ducati", "Harley Davidson"]

DUNGTICH_MIDPOINT = {
    "Dưới 50 cc": 35,
    "50 - 100 cc": 75,
    "100 - 175 cc": 137.5,
    "Trên 175 cc": 250,
    "Không biết rõ": np.nan,
}

NUMERIC_FEATURES_P2 = [
    "TuoiXe",
    "SoKm",
    "Km_per_year",
    "Km_unknown",
    "DongXe_freq",
    "pct_ngoai_hinh",
    "co_pct_ngoai_hinh",
    "co_chinh_chu",
    "co_bstp",
    "da_sang_ten",
    "con_bao_hanh",
    "khong_dam_dung",
    "do_dai_mota",
    "Dung_tich_so",
    "log_SoKm",
    "XuatXu_khong_ro",
    "nhap_khau",
    "co_do_choi",
    "mot_doi_chu",
    "cong_chung",
    "uy_quyen",
    "con_moi_dep",
    "giay_to_day_du",
    "bao_tranh_chap",
    "thanh_ly_gap",
    "cam_ket_khong_ngap",
    "co_abs",
    "it_su_dung",
    "tra_gop",
    "xe_zin_nguyen_ban",
    "thuong_luong",
    "xe_grab_dich_vu",
]

CATEGORICAL_FEATURES_P2 = [
    "Thương hiệu",
    "Loại xe",
    "Dung tích xe",
    "Xuất xứ",
    "Quan",
    "PhanKhucXe",
    "SoKm_Group",
    "SoKm_Year",
    "Tuoi_xe_group",
]

MODEL_INPUT_FEATURES_P2 = (
    NUMERIC_FEATURES_P2
    + CATEGORICAL_FEATURES_P2
    + ["Dòng xe"]
)


# ============================================================================
# ALIAS ĐẦU VÀO
# ============================================================================

COLUMN_ALIASES = {
    # Thương hiệu
    "thuong_hieu": "Thương hiệu",
    "thương_hiệu": "Thương hiệu",
    "thuong hieu": "Thương hiệu",
    "brand": "Thương hiệu",

    # Dòng xe
    "dong_xe": "Dòng xe",
    "dòng_xe": "Dòng xe",
    "dong xe": "Dòng xe",
    "model": "Dòng xe",
    "model_name": "Dòng xe",

    # Loại xe
    "loai_xe": "Loại xe",
    "loại_xe": "Loại xe",
    "loai xe": "Loại xe",
    "vehicle_type": "Loại xe",

    # Dung tích xe
    "dung_tich_xe": "Dung tích xe",
    "dung tích xe": "Dung tích xe",
    "dung_tich": "Dung tích xe",
    "engine_size": "Dung tích xe",
    "capacity": "Dung tích xe",

    # Xuất xứ
    "xuat_xu": "Xuất xứ",
    "xuất_xứ": "Xuất xứ",
    "xuat xu": "Xuất xứ",
    "origin": "Xuất xứ",

    # Năm đăng ký
    "nam_dang_ky": "Năm đăng ký",
    "năm_đăng_ký": "Năm đăng ký",
    "nam dang ky": "Năm đăng ký",
    "registration_year": "Năm đăng ký",
    "year": "Năm đăng ký",

    # Km
    "so_km": "Số Km đã đi",
    "số_km": "Số Km đã đi",
    "so km": "Số Km đã đi",
    "so_km_da_di": "Số Km đã đi",
    "kilometers": "Số Km đã đi",
    "kilometres": "Số Km đã đi",
    "mileage": "Số Km đã đi",
    "km": "Số Km đã đi",

    # Quận / địa chỉ
    "quan": "Quan",
    "quận": "Quan",
    "district": "Quan",
    "quan_huyen": "Quan",
    "quận_huyện": "Quan",

    "dia_chi": "Địa chỉ",
    "địa_chỉ": "Địa chỉ",
    "address": "Địa chỉ",

    # Text
    "tieu_de": "Tiêu đề",
    "tiêu_đề": "Tiêu đề",
    "title": "Tiêu đề",

    "mo_ta": "Mô tả chi tiết",
    "mô_tả": "Mô tả chi tiết",
    "mo_ta_chi_tiet": "Mô tả chi tiết",
    "description": "Mô tả chi tiết",

    # Giá
    "gia": "Giá",
    "giá": "Giá",
    "gia_rao": "Giá rao",
    "giá_rao": "Giá rao",
    "gia ban": "Giá rao",
    "asking_price": "Giá rao",
    "price": "Giá rao",

    # Năm tham chiếu
    "reference_year": "reference_year",
    "nam_tham_chieu": "reference_year",
}

CANONICAL_COLUMNS = [
    "Thương hiệu",
    "Dòng xe",
    "Loại xe",
    "Dung tích xe",
    "Xuất xứ",
    "Năm đăng ký",
    "Số Km đã đi",
    "Quan",
    "Địa chỉ",
    "Tiêu đề",
    "Mô tả chi tiết",
    "Giá",
    "Giá rao",
    "reference_year",
]


# ============================================================================
# HÀM HỖ TRỢ CHUẨN HÓA
# ============================================================================

def _to_dataframe(
    data: Mapping[str, Any] | pd.Series | pd.DataFrame,
) -> pd.DataFrame:
    """Chuyển dict/Series/DataFrame thành một DataFrame độc lập."""
    if isinstance(data, pd.DataFrame):
        return data.copy(deep=True)

    if isinstance(data, pd.Series):
        return data.to_frame().T.copy(deep=True)

    if isinstance(data, Mapping):
        return pd.DataFrame([dict(data)])

    raise TypeError(
        "Dữ liệu đầu vào phải là Mapping, pandas Series hoặc pandas DataFrame."
    )


def _normalized_column_key(value: Any) -> str:
    key = str(value).strip().lower()
    key = re.sub(r"\s+", " ", key)
    return key


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    canonical_lookup = {
        _normalized_column_key(column): column
        for column in CANONICAL_COLUMNS
    }

    rename_map: dict[Any, str] = {}

    for column in result.columns:
        key = _normalized_column_key(column)

        if key in canonical_lookup:
            rename_map[column] = canonical_lookup[key]
        elif key in COLUMN_ALIASES:
            rename_map[column] = COLUMN_ALIASES[key]

    result = result.rename(columns=rename_map)

    # Nếu alias tạo cột trùng tên, lấy giá trị không rỗng đầu tiên từ trái sang phải.
    if result.columns.duplicated().any():
        merged: dict[str, pd.Series] = {}

        for column in result.columns.unique():
            subset = result.loc[:, result.columns == column]

            if subset.shape[1] == 1:
                merged[column] = subset.iloc[:, 0]
            else:
                merged[column] = (
                    subset
                    .replace("", np.nan)
                    .bfill(axis=1)
                    .iloc[:, 0]
                )

        result = pd.DataFrame(merged, index=result.index)

    return result


def _clean_category(value: Any, default: str) -> str:
    if pd.isna(value):
        return default

    text = re.sub(r"\s+", " ", str(value).strip())

    if text.lower() in {"", "nan", "none", "null", "n/a", "na"}:
        return default

    return text


def _clean_raw_text(value: Any) -> str:
    if pd.isna(value):
        return ""

    return re.sub(r"\s+", " ", str(value).strip())


def _reference_year(value: Any) -> int:
    if pd.isna(value):
        return REFERENCE_YEAR_P2

    try:
        year = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return REFERENCE_YEAR_P2

    if year < 1950 or year > 2100:
        return REFERENCE_YEAR_P2

    return year


def _safe_numeric_median(series: pd.Series, fallback: float = 0.0) -> float:
    values = pd.to_numeric(series, errors="coerce")
    median = values.median()

    if pd.isna(median):
        return float(fallback)

    return float(median)


def _first_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column

    return None


# ============================================================================
# HELPER GIỐNG NOTEBOOK P2
# ============================================================================

def extract_pct_ngoai_hinh(text: Any) -> float:
    """
    Trích phần trăm ngoại hình theo đúng regex notebook P2.
    """
    if pd.isna(text):
        return np.nan

    match = re.search(r"(\d{2,3})\s*%", str(text))

    if match:
        value = int(match.group(1))
        return float(value) if value <= 100 else np.nan

    return np.nan


def contains_keyword(text: Any, keywords: list[str]) -> int:
    """
    Logic đúng notebook P2: lowercase text và kiểm tra substring.
    """
    if pd.isna(text):
        return 0

    text_lower = str(text).lower()
    return int(any(keyword in text_lower for keyword in keywords))


def flag_keyword(text: Any, patterns: list[str]) -> int:
    """
    Logic đúng notebook P2: trả về 1 nếu khớp ít nhất một regex.
    """
    if pd.isna(text):
        return 0

    value = str(text).lower()
    return int(any(re.search(pattern, value) for pattern in patterns))


def map_segment(row: pd.Series) -> str:
    """
    Ánh xạ phân khúc theo thương hiệu/dòng xe như notebook P2.

    Notebook chuyển Dòng xe sang lowercase và chọn keyword dài nhất.
    Việc lowercase keyword lúc so sánh chỉ xử lý nhất quán mục "Air Blade",
    không thay đổi kết quả mong muốn của bảng ánh xạ P2.
    """
    brand = str(row.get("Thương hiệu", UNKNOWN_CATEGORY))
    model_name = str(row.get("Dòng xe", UNKNOWN_CATEGORY)).lower()

    if brand in SEGMENT_MAP:
        all_matches: list[tuple[str, str]] = []

        for tier, keywords in SEGMENT_MAP[brand].items():
            for keyword in keywords:
                keyword_lower = str(keyword).lower()

                if keyword_lower in model_name:
                    all_matches.append((keyword_lower, tier))

        if all_matches:
            _, best_tier = max(all_matches, key=lambda item: len(item[0]))
            return best_tier

    if brand in MIXED_MOTO_BRANDS:
        return "moto cao cấp"

    if brand in LUXURY_MOTO_BRANDS:
        return "moto siêu sang"

    return "trung cấp"


# ============================================================================
# NORMALIZE INPUT
# ============================================================================

def normalize_input(
    data: Mapping[str, Any] | pd.Series | pd.DataFrame,
    reference_year: int | None = None,
) -> pd.DataFrame:
    """
    Chuẩn hóa đầu vào của app hoặc dữ liệu batch thành cấu trúc P2.

    Các cột quan trọng được sinh:
    - NamDangKy
    - Km_clean
    - Km_unknown
    - SoKm
    - Quan
    - Tieu_de_clean
    - Mo_ta_clean
    - Text_combined
    - Gia_clean / Gia_rao_clean
    """
    df = _to_dataframe(data)

    if df.empty:
        raise ValueError("Dữ liệu đầu vào không có bản ghi.")

    df = _rename_columns(df)

    defaults = {
        "Thương hiệu": UNKNOWN_CATEGORY,
        "Dòng xe": UNKNOWN_CATEGORY,
        "Loại xe": UNKNOWN_CATEGORY,
        "Dung tích xe": DEFAULT_ENGINE_SIZE,
        "Xuất xứ": DEFAULT_ORIGIN,
        "Năm đăng ký": np.nan,
        "Số Km đã đi": UNKNOWN_KM,
        "Quan": UNKNOWN_CATEGORY,
        "Địa chỉ": UNKNOWN_CATEGORY,
        "Tiêu đề": "",
        "Mô tả chi tiết": "",
        "Giá": np.nan,
        "Giá rao": np.nan,
        "reference_year": (
            REFERENCE_YEAR_P2 if reference_year is None else reference_year
        ),
    }

    for column, default_value in defaults.items():
        if column not in df.columns:
            df[column] = default_value

    category_defaults = {
        "Thương hiệu": UNKNOWN_CATEGORY,
        "Dòng xe": UNKNOWN_CATEGORY,
        "Loại xe": UNKNOWN_CATEGORY,
        "Dung tích xe": DEFAULT_ENGINE_SIZE,
        "Xuất xứ": DEFAULT_ORIGIN,
    }

    for column, default_value in category_defaults.items():
        df[column] = df[column].apply(
            lambda value, default=default_value: _clean_category(value, default)
        )

    # Năm tham chiếu: app.py truyền 2025, đúng notebook P2.
    if reference_year is not None:
        df["reference_year"] = _reference_year(reference_year)
    else:
        df["reference_year"] = (
            df["reference_year"]
            .apply(_reference_year)
            .astype(int)
        )

    # Notebook dùng NamDangKy.
    if "NamDangKy" in df.columns:
        parsed_existing_year = pd.to_numeric(df["NamDangKy"], errors="coerce")
        parsed_input_year = df["Năm đăng ký"].apply(parse_year)
        df["NamDangKy"] = parsed_existing_year.fillna(parsed_input_year)
    elif "Nam_clean" in df.columns:
        parsed_existing_year = pd.to_numeric(df["Nam_clean"], errors="coerce")
        parsed_input_year = df["Năm đăng ký"].apply(parse_year)
        df["NamDangKy"] = parsed_existing_year.fillna(parsed_input_year)
    else:
        df["NamDangKy"] = df["Năm đăng ký"].apply(parse_year)

    df["NamDangKy"] = pd.to_numeric(df["NamDangKy"], errors="coerce")
    df["Nam_clean"] = df["NamDangKy"]
    df["Năm đăng ký"] = df["NamDangKy"]

    # Notebook: 999999 là "không rõ"; SoKm đổi thành NaN.
    parsed_km = df["Số Km đã đi"].apply(parse_km)
    parsed_km = pd.to_numeric(parsed_km, errors="coerce").fillna(UNKNOWN_KM)
    parsed_km = parsed_km.clip(lower=0)

    df["Km_clean"] = parsed_km.astype(int)
    df["Số Km đã đi"] = df["Km_clean"]
    df["Km_unknown"] = (df["Số Km đã đi"] == UNKNOWN_KM).astype(int)
    df["SoKm"] = df["Số Km đã đi"].where(
        df["Số Km đã đi"] != UNKNOWN_KM,
        np.nan,
    )

    # Quận: ưu tiên Quan do form app đã cung cấp; nếu không có thì tách từ Địa chỉ.
    quan_value = df["Quan"].apply(
        lambda value: _clean_category(value, UNKNOWN_CATEGORY)
    )
    address_value = df["Địa chỉ"].apply(extract_district).apply(
        lambda value: _clean_category(value, UNKNOWN_CATEGORY)
    )

    invalid_quan = {
        "", "không rõ", "nan", "none", "null", "n/a", "na"
    }

    use_address = quan_value.str.lower().isin(invalid_quan)
    df["Quan"] = quan_value.where(~use_address, address_value)
    df["Quan"] = df["Quan"].replace("", UNKNOWN_CATEGORY)
    df["Quan_clean"] = df["Quan"]

    df["Địa chỉ"] = df["Địa chỉ"].apply(
        lambda value: _clean_category(value, UNKNOWN_CATEGORY)
    )

    # Notebook dùng text gốc; các helper tự lowercase.
    df["Tiêu đề"] = df["Tiêu đề"].apply(_clean_raw_text)
    df["Mô tả chi tiết"] = df["Mô tả chi tiết"].apply(_clean_raw_text)

    df["Tieu_de_clean"] = df["Tiêu đề"].apply(normalize_text)
    df["Mo_ta_clean"] = df["Mô tả chi tiết"].apply(normalize_text)

    df["Text_combined"] = (
        df["Tiêu đề"].fillna("")
        + " "
        + df["Mô tả chi tiết"].fillna("")
    )

    # Hỗ trợ cả dữ liệu raw (Giá), dữ liệu đã clean (Giá_clean),
    # và input app (Giá rao).
    if "Giá_clean" in df.columns:
        gia_clean = pd.to_numeric(df["Giá_clean"], errors="coerce")
    elif "Gia_clean" in df.columns:
        gia_clean = pd.to_numeric(df["Gia_clean"], errors="coerce")
    else:
        gia_clean = df["Giá"].apply(parse_money_to_million)

    gia_rao_clean = df["Giá rao"].apply(parse_money_to_million)

    df["Gia_clean"] = gia_clean
    df["Gia_rao_clean"] = gia_rao_clean

    return df.reset_index(drop=True)


# ============================================================================
# CÁC BƯỚC FEATURE ENGINEERING
# ============================================================================

def create_age_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Notebook P2:
        TuoiXe = 2025 - NamDangKy

    App truyền reference_year=2025 nên giữ đúng kết quả notebook.
    """
    result = df.copy()

    if "NamDangKy" not in result.columns:
        result["NamDangKy"] = result["Năm đăng ký"].apply(parse_year)

    years = pd.to_numeric(result["NamDangKy"], errors="coerce")

    # Để bám sát P2, mặc định dùng 2025; nếu app đã truyền cột reference_year,
    # giá trị đó được dùng và app hiện cấu hình đúng 2025.
    if "reference_year" in result.columns:
        reference = pd.to_numeric(
            result["reference_year"],
            errors="coerce",
        ).fillna(REFERENCE_YEAR_P2)
    else:
        reference = pd.Series(
            REFERENCE_YEAR_P2,
            index=result.index,
            dtype=float,
        )

    result["TuoiXe"] = reference - years

    return result


def create_km_per_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Notebook P2:
        Km_per_year = SoKm / (TuoiXe + 1)
    """
    result = df.copy()

    if "SoKm" not in result.columns:
        result["SoKm"] = result["Số Km đã đi"].where(
            result["Số Km đã đi"] != UNKNOWN_KM,
            np.nan,
        )

    if "TuoiXe" not in result.columns:
        result = create_age_feature(result)

    result["SoKm"] = pd.to_numeric(result["SoKm"], errors="coerce")
    result["TuoiXe"] = pd.to_numeric(result["TuoiXe"], errors="coerce")

    result["Km_per_year"] = (
        result["SoKm"] / (result["TuoiXe"] + 1)
    )

    return result


def create_text_length_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo text_full và độ dài mô tả đúng notebook P2.
    """
    result = df.copy()

    result["text_full"] = (
        result["Tiêu đề"].fillna("")
        + " "
        + result["Mô tả chi tiết"].fillna("")
    )

    result["do_dai_mota"] = (
        result["text_full"]
        .fillna("")
        .str.len()
    )

    return result


def create_keyword_features(
    df: pd.DataFrame,
    pct_median: float | None = None,
) -> pd.DataFrame:
    """
    Tạo toàn bộ text/keyword feature theo đúng notebook P2.
    """
    result = df.copy()

    if "text_full" not in result.columns:
        result = create_text_length_features(result)

    # % ngoại hình
    result["pct_ngoai_hinh"] = (
        result["text_full"]
        .apply(extract_pct_ngoai_hinh)
    )

    result["co_pct_ngoai_hinh"] = (
        result["pct_ngoai_hinh"]
        .notna()
        .astype(int)
    )

    if pct_median is None:
        pct_median = _safe_numeric_median(
            result["pct_ngoai_hinh"],
            fallback=0.0,
        )

    result["pct_ngoai_hinh"] = (
        result["pct_ngoai_hinh"]
        .fillna(float(pct_median))
    )

    # Bộ cũ
    result["co_chinh_chu"] = result["text_full"].apply(
        lambda text: contains_keyword(text, ["chính chủ"])
    )
    result["co_bstp"] = result["text_full"].apply(
        lambda text: contains_keyword(text, ["bstp"])
    )
    result["da_sang_ten"] = result["text_full"].apply(
        lambda text: contains_keyword(text, ["sang tên"])
    )
    result["con_bao_hanh"] = result["text_full"].apply(
        lambda text: contains_keyword(text, ["còn bảo hành"])
    )
    result["khong_dam_dung"] = result["text_full"].apply(
        lambda text: contains_keyword(
            text,
            ["không đâm đụng", "chưa đâm đụng", "ko đâm đụng"],
        )
    )

    # Bộ bổ sung
    result["nhap_khau"] = result["text_full"].apply(
        lambda text: flag_keyword(
            text,
            [
                r"nhập khẩu",
                r"nhập ý",
                r"nhập thái",
                r"nhập nhật",
                r"nhập đức",
            ],
        )
    )

    result["co_do_choi"] = result["text_full"].apply(
        lambda text: flag_keyword(
            text,
            [
                r"đồ chơi",
                r"độ full",
                r"lên đồ",
                r"dàn đồ chơi",
            ],
        )
    )

    result["mot_doi_chu"] = result["text_full"].apply(
        lambda text: flag_keyword(
            text,
            [
                r"1 đời chủ",
                r"một đời chủ",
                r"chủ đầu",
                r"từ đầu đến giờ",
            ],
        )
    )

    result["cong_chung"] = result["text_full"].apply(
        lambda text: flag_keyword(text, [r"công chứng"])
    )

    result["uy_quyen"] = result["text_full"].apply(
        lambda text: flag_keyword(
            text,
            [r"ủy quyền", r"uỷ quyền"],
        )
    )

    result["con_moi_dep"] = result["text_full"].apply(
        lambda text: flag_keyword(
            text,
            [
                r"còn mới",
                r"như mới",
                r"mới cứng",
                r"cực đẹp",
                r"siêu đẹp",
            ],
        )
    )

    result["giay_to_day_du"] = result["text_full"].apply(
        lambda text: flag_keyword(
            text,
            [
                r"giấy tờ đầy đủ",
                r"giấy tờ hợp lệ",
                r"đầy đủ giấy tờ",
            ],
        )
    )

    result["bao_tranh_chap"] = result["text_full"].apply(
        lambda text: flag_keyword(
            text,
            [
                r"bao tranh chấp",
                r"bao test",
                r"bao check",
            ],
        )
    )

    result["thanh_ly_gap"] = result["text_full"].apply(
        lambda text: flag_keyword(
            text,
            [
                r"thanh lý",
                r"cần bán gấp",
                r"bán gấp",
                r"cần tiền",
            ],
        )
    )

    # Bộ feature mới của notebook P2
    result["cam_ket_khong_ngap"] = result["text_full"].apply(
        lambda text: flag_keyword(
            text,
            [
                r"chưa từng ngập",
                r"không từng ngập",
                r"ko từng ngập",
                r"không ngập",
                r"ko ngập",
                r"chưa ngập",
            ],
        )
    )

    result["co_abs"] = result["text_full"].apply(
        lambda text: flag_keyword(text, [r"\babs\b"])
    )

    result["it_su_dung"] = result["text_full"].apply(
        lambda text: flag_keyword(
            text,
            [
                r"ít sử dụng",
                r"đi ít",
                r"ít đi",
            ],
        )
    )

    result["tra_gop"] = result["text_full"].apply(
        lambda text: flag_keyword(text, [r"trả góp"])
    )

    result["xe_zin_nguyen_ban"] = result["text_full"].apply(
        lambda text: flag_keyword(
            text,
            [
                r"\bzin\b",
                r"nguyên bản",
                r"nguyên zin",
            ],
        )
    )

    result["thuong_luong"] = result["text_full"].apply(
        lambda text: flag_keyword(
            text,
            [
                r"thương lượng",
                r"\btl\b",
                r"trao đổi",
            ],
        )
    )

    result["xe_grab_dich_vu"] = result["text_full"].apply(
        lambda text: flag_keyword(
            text,
            [
                r"\bgrab\b",
                r"chạy dịch vụ",
                r"chạy xe ôm",
            ],
        )
    )

    return result


# ============================================================================
# THỐNG KÊ TỪ REFERENCE DATA
# ============================================================================

def _prepare_reference_statistics(
    reference_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Tái tạo các thống kê mà notebook P2 tính trên toàn bộ DataFrame:
    - value_counts của Dòng xe
    - median pct_ngoai_hinh
    - median Dung_tich_so
    - median SoKm
    """
    reference = normalize_input(
        reference_df,
        reference_year=REFERENCE_YEAR_P2,
    )

    reference["text_full"] = (
        reference["Tiêu đề"].fillna("")
        + " "
        + reference["Mô tả chi tiết"].fillna("")
    )

    model_frequency = reference["Dòng xe"].value_counts()

    pct_values = reference["text_full"].apply(extract_pct_ngoai_hinh)
    pct_median = _safe_numeric_median(pct_values, fallback=0.0)

    engine_values = reference["Dung tích xe"].map(DUNGTICH_MIDPOINT)
    engine_median = _safe_numeric_median(
        engine_values,
        fallback=137.5,
    )

    km_values = pd.to_numeric(reference["SoKm"], errors="coerce")
    km_median = _safe_numeric_median(km_values, fallback=0.0)

    return {
        "reference": reference,
        "dongxe_freq": model_frequency,
        "pct_median": pct_median,
        "engine_median": engine_median,
        "km_median": km_median,
    }


# ============================================================================
# HÀM CHÍNH DÙNG BỞI APP.PY
# ============================================================================

def build_input_dataframe(
    prediction_values: Mapping[str, Any] | pd.Series | pd.DataFrame,
    reference_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Tạo DataFrame đầu vào cho pipeline dự đoán.

    Thứ tự:
    1. normalize_input
    2. create_age_feature
    3. create_km_per_year
    4. create_text_length_features
    5. create_keyword_features
    6. Các feature số/categorical còn lại của notebook P2

    Parameters
    ----------
    prediction_values:
        Dict từ form Streamlit, Series hoặc DataFrame batch.

    reference_df:
        Dữ liệu tham chiếu/training. Bắt buộc để tái tạo đúng
        DongXe_freq và các median của notebook P2.

    Returns
    -------
    pandas.DataFrame
        DataFrame chứa đầy đủ feature của notebook P2.
    """
    if reference_df is None:
        raise ValueError(
            "reference_df là bắt buộc để tái tạo DongXe_freq "
            "và các median giống notebook P2."
        )

    stats = _prepare_reference_statistics(reference_df)

    df = normalize_input(
        prediction_values,
        reference_year=REFERENCE_YEAR_P2,
    )

    # PhanKhucXe trong notebook được tạo trước feature_engineering(df).
    df["PhanKhucXe"] = df.apply(map_segment, axis=1)

    df = create_age_feature(df)
    df = create_km_per_year(df)
    df = create_text_length_features(df)
    df = create_keyword_features(
        df,
        pct_median=stats["pct_median"],
    )

    # Đặc trưng số cơ bản
    df["PhanKhuc"] = (
        df["Thương hiệu"].astype(str)
        + " | "
        + df["Dung tích xe"].astype(str)
    )

    df["DongXe_freq"] = (
        df["Dòng xe"]
        .map(stats["dongxe_freq"])
        .fillna(0)
    )

    # Notebook tạo log_gia từ Giá_clean. Khi dự đoán giá mới có thể thiếu target.
    if "Gia_clean" in df.columns:
        df["log_gia"] = np.log1p(
            pd.to_numeric(df["Gia_clean"], errors="coerce")
        )
    else:
        df["log_gia"] = np.nan

    # Dung tích số hóa
    df["Dung_tich_so"] = (
        df["Dung tích xe"]
        .map(DUNGTICH_MIDPOINT)
        .fillna(stats["engine_median"])
    )

    # log số km
    df["log_SoKm"] = np.log1p(
        pd.to_numeric(df["SoKm"], errors="coerce")
        .fillna(stats["km_median"])
    )

    df["XuatXu_khong_ro"] = (
        df["Xuất xứ"] == "Đang cập nhật"
    ).astype(int)

    # Các nhóm pd.cut đúng notebook P2
    bins_sokm = [0, 10000, 25000, 50000, np.inf]
    labels_sokm = ["<10k", "10k-25k", "25k-50k", ">50k"]

    df["SoKm_Group"] = pd.cut(
        df["Số Km đã đi"],
        bins=bins_sokm,
        labels=labels_sokm,
        include_lowest=True,
        right=False,
    )

    bins_sokm_year = [0, 1000, 2000, 5000, np.inf]
    labels_sokm_year = ["<1k", "1k-2k", "2k-5k", ">5k"]

    df["SoKm_Year"] = pd.cut(
        df["Km_per_year"],
        bins=bins_sokm_year,
        labels=labels_sokm_year,
        include_lowest=True,
        right=False,
    )

    bins_tuoi_xe = [0, 3, 6, 10, 15, np.inf]
    labels_tuoi_xe = ["1-3", "3-6", "6-10", "10-15", ">15"]

    df["Tuoi_xe_group"] = pd.cut(
        df["TuoiXe"],
        bins=bins_tuoi_xe,
        labels=labels_tuoi_xe,
        include_lowest=True,
        right=False,
    )

    df["PhanKhuc_KetHop"] = (
        df["PhanKhucXe"].astype(str)
        + "__"
        + df["Tuoi_xe_group"].astype(str)
        + "__"
        + df["SoKm_Year"].astype(str)
    )

    # Notebook drop text_full ở cuối.
    df = df.drop(columns=["text_full"], errors="ignore")

    return df.reset_index(drop=True)


# ============================================================================
# HÀM PORT TRỰC TIẾP CHO DATAFRAME ĐÃ TIỀN XỬ LÝ
# ============================================================================

def feature_engineering(
    df: pd.DataFrame,
    reference_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Phiên bản module hóa của feature_engineering(df) trong notebook P2.

    - Nếu reference_df không truyền, dùng chính df để tính frequency/median,
      giống notebook P2.
    - Nếu dự đoán một/batch mới, nên truyền reference_df là dữ liệu train/raw.
    """
    reference = df if reference_df is None else reference_df

    return build_input_dataframe(
        prediction_values=df,
        reference_df=reference,
    )


def select_model_input_features(
    df: pd.DataFrame,
    strict: bool = True,
) -> pd.DataFrame:
    """
    Trả về đúng các cột X mà notebook P2 dùng:
        numeric_features + categorical_features + ['Dòng xe']

    Pipeline sklearn thường tự chọn cột nên app không bắt buộc gọi hàm này.
    """
    missing = [
        column
        for column in MODEL_INPUT_FEATURES_P2
        if column not in df.columns
    ]

    if missing and strict:
        raise ValueError(
            "Thiếu các feature đầu vào của notebook P2: "
            + ", ".join(missing)
        )

    available = [
        column
        for column in MODEL_INPUT_FEATURES_P2
        if column in df.columns
    ]

    return df[available].copy()


def validate_p2_features(df: pd.DataFrame) -> list[str]:
    """
    Trả về danh sách feature P2 còn thiếu.
    Danh sách rỗng nghĩa là DataFrame đã đủ đầu vào cho model.
    """
    return [
        column
        for column in MODEL_INPUT_FEATURES_P2
        if column not in df.columns
    ]
