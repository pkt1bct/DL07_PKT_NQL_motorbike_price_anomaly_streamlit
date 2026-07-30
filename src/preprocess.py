# ==========================================================
# src/preprocess.py
# ==========================================================
# Các hàm tiền xử lý dữ liệu
# Dùng chung cho:
#   - EDA
#   - Prediction
#   - Anomaly Detection
#   - Batch Prediction
# ==========================================================

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ==========================================================
# HẰNG SỐ
# ==========================================================

CURRENT_YEAR = 2025

UNKNOWN_KM = 999999

MIN_YEAR = 1950

MAX_YEAR = CURRENT_YEAR

PRICE_COLUMNS = [
    "Giá",
    "Giá bán",
    "Giá rao",
    "price",
]

DISTRICT_COLUMNS = [
    "Địa chỉ",
    "Quan",
    "Quận",
]

TEXT_COLUMNS = [
    "Tiêu đề",
    "Mô tả chi tiết",
    "Mô tả",
]

REQUIRED_BATCH_COLUMNS = [
    "thuong_hieu",
    "dong_xe",
    "loai_xe",
    "nam_dang_ky",
    "gia_rao",
]

OPTIONAL_BATCH_COLUMNS = [
    "so_km",
    "dung_tich_xe",
    "xuat_xu",
    "quan",
    "tieu_de",
    "mo_ta",
]

# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """
    Đọc dataset từ Excel hoặc CSV.

    Parameters
    ----------
    file_path

    Returns
    -------
    DataFrame
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    suffix = file_path.suffix.lower()

    if suffix == ".xlsx":
        df = pd.read_excel(file_path)

    elif suffix == ".csv":
        df = pd.read_csv(file_path)

    else:
        raise ValueError(
            "Chỉ hỗ trợ Excel hoặc CSV."
        )

    return df.copy()


# ==========================================================
# CHUẨN HÓA TEXT
# ==========================================================

def normalize_text(text: Any) -> str:
    """
    Chuẩn hóa chuỗi.

    - bỏ khoảng trắng
    - unicode NFC
    - lower
    """

    if pd.isna(text):
        return ""

    text = str(text)

    text = unicodedata.normalize("NFC", text)

    text = text.strip()

    text = re.sub(r"\s+", " ", text)

    return text.lower()


# ==========================================================
# BỎ DẤU TIẾNG VIỆT
# ==========================================================

def remove_accent(text: Any) -> str:

    if pd.isna(text):
        return ""

    text = str(text)

    text = unicodedata.normalize("NFD", text)

    text = "".join(
        c for c in text
        if unicodedata.category(c) != "Mn"
    )

    return text.lower()


# ==========================================================
# PARSE MONEY
# ==========================================================

def parse_money_to_million(value: Any) -> float:
    """
    Chuyển giá về đơn vị TRIỆU ĐỒNG.

    Ví dụ

    38 triệu

    38tr

    38.5

    38,5

    38.500.000

    38500000

    đều trả về

    38.5
    """

    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float)):

        value = float(value)

        if value > 100000:

            return value / 1_000_000

        return value

    text = normalize_text(value)

    text = text.replace("vnđ", "")

    text = text.replace("vnd", "")

    text = text.replace("đ", "")

    text = text.replace("dong", "")

    text = text.replace("triệu", "")

    text = text.replace("trieu", "")

    text = text.replace("tr", "")

    text = text.replace(",", ".")

    text = text.replace(" ", "")

    try:

        number = float(text)

    except:

        digits = re.findall(r"\d+", text)

        if len(digits) == 0:
            return np.nan

        number = float("".join(digits))

    if number > 100000:

        number = number / 1_000_000

    return round(number, 2)


# ==========================================================
# PARSE YEAR
# ==========================================================

def parse_year(value: Any):
    """
    Chuẩn hóa năm đăng ký.

    Ví dụ

    2019

    2020

    Trước năm 1980

    1985

    """

    if pd.isna(value):
        return np.nan

    text = normalize_text(value)

    if "1980" in text:

        return 1979

    digits = re.findall(r"\d{4}", text)

    if len(digits) == 0:
        return np.nan

    year = int(digits[0])

    if year < MIN_YEAR:
        return np.nan

    if year > MAX_YEAR:
        return np.nan

    return year


# ==========================================================
# PARSE KM
# ==========================================================

def parse_km(value: Any):
    """
    Chuẩn hóa số Km.

    Ví dụ

    12000

    12.000 km

    12,000

    Không rõ

    """

    if pd.isna(value):
        return UNKNOWN_KM

    if isinstance(value, (int, float)):

        return int(value)

    text = normalize_text(value)

    if "không" in text:

        return UNKNOWN_KM

    text = text.replace("km", "")

    text = text.replace(".", "")

    text = text.replace(",", "")

    digits = re.findall(r"\d+", text)

    if len(digits) == 0:

        return UNKNOWN_KM

    return int("".join(digits))


# ==========================================================
# TÁCH QUẬN / HUYỆN
# ==========================================================

def extract_district(address: Any) -> str:
    """
    Lấy Quận/Huyện từ địa chỉ.

    Ví dụ

    Quận 10, TP.HCM

    => Quận 10

    Bình Thạnh, TP.HCM

    => Bình Thạnh
    """

    if pd.isna(address):
        return "Không rõ"

    address = str(address)

    address = address.strip()

    if "," in address:

        return address.split(",")[0].strip()

    return address

# PHẦN 2 - CLEAN DATAFRAME

# ==========================================================
# TÌM CỘT ĐẦU TIÊN TỒN TẠI
# ==========================================================

def find_existing_column(
    df: pd.DataFrame,
    candidates: list[str]
) -> str | None:
    """
    Trả về tên cột đầu tiên tồn tại trong DataFrame.
    """

    for col in candidates:

        if col in df.columns:
            return col

    return None


# ==========================================================
# CLEAN PRICE COLUMN
# ==========================================================

def create_clean_price_column(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Chuẩn hóa cột Giá.

    Tạo thêm cột

        Gia_clean

    đơn vị:

        triệu đồng
    """

    df = df.copy()

    price_col = find_existing_column(
        df,
        PRICE_COLUMNS
    )

    if price_col is None:
        raise ValueError(
            "Không tìm thấy cột giá."
        )

    df["Gia_clean"] = (
        df[price_col]
        .apply(parse_money_to_million)
    )

    return df


# ==========================================================
# CLEAN YEAR COLUMN
# ==========================================================

def create_clean_year_column(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Chuẩn hóa năm đăng ký.

    Tạo

        Nam_clean

        Tuoi_xe
    """

    df = df.copy()

    year_col = None

    for c in [
        "Năm đăng ký",
        "Nam dang ky",
        "nam_dang_ky",
    ]:

        if c in df.columns:
            year_col = c
            break

    if year_col is None:

        return df

    df["Nam_clean"] = (
        df[year_col]
        .apply(parse_year)
    )

    df["Tuoi_xe"] = (
        CURRENT_YEAR
        - df["Nam_clean"]
    )

    return df


# ==========================================================
# CLEAN KM
# ==========================================================

def create_clean_km_column(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Chuẩn hóa số Km.

    Tạo

        Km_clean
    """

    df = df.copy()

    km_col = None

    for c in [

        "Số Km đã đi",

        "So Km da di",

        "so_km",

    ]:

        if c in df.columns:

            km_col = c

            break

    if km_col is None:

        df["Km_clean"] = UNKNOWN_KM

        return df

    df["Km_clean"] = (
        df[km_col]
        .apply(parse_km)
    )

    return df


# ==========================================================
# CLEAN DISTRICT
# ==========================================================

def create_clean_district_column(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Chuẩn hóa Quận/Huyện.

    Sinh cột

        Quan_clean
    """

    df = df.copy()

    district_col = find_existing_column(
        df,
        DISTRICT_COLUMNS
    )

    if district_col is None:

        df["Quan_clean"] = "Không rõ"

        return df

    df["Quan_clean"] = (
        df[district_col]
        .apply(extract_district)
    )

    return df


# ==========================================================
# CLEAN TEXT
# ==========================================================

def create_clean_text_columns(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Chuẩn hóa text.

    Sinh thêm

        Tieu_de_clean

        Mo_ta_clean
    """

    df = df.copy()

    if "Tiêu đề" in df.columns:

        df["Tieu_de_clean"] = (
            df["Tiêu đề"]
            .apply(normalize_text)
        )

    if "Mô tả chi tiết" in df.columns:

        df["Mo_ta_clean"] = (
            df["Mô tả chi tiết"]
            .apply(normalize_text)
        )

    return df


# ==========================================================
# CLEAN CATEGORICAL
# ==========================================================

def clean_categorical_columns(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Chuẩn hóa các biến phân loại.
    """

    df = df.copy()

    cols = [

        "Thương hiệu",

        "Dòng xe",

        "Loại xe",

        "Dung tích xe",

        "Xuất xứ",

    ]

    for col in cols:

        if col in df.columns:

            df[col] = (

                df[col]

                .astype(str)

                .str.strip()

            )

    return df


# ==========================================================
# DROP DUPLICATE
# ==========================================================

def remove_duplicate_rows(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Xóa dòng trùng.
    """

    return (
        df
        .drop_duplicates()
        .reset_index(drop=True)
    )


# ==========================================================
# DROP INVALID PRICE
# ==========================================================

def remove_invalid_price(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Loại bỏ dòng giá không hợp lệ.
    """

    df = df.copy()

    if "Gia_clean" not in df.columns:

        return df

    df = df[
        df["Gia_clean"].notna()
    ]

    df = df[
        df["Gia_clean"] > 0
    ]

    return df.reset_index(drop=True)


# ==========================================================
# DROP INVALID YEAR
# ==========================================================

def remove_invalid_year(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Loại bỏ năm không hợp lệ.
    """

    df = df.copy()

    if "Nam_clean" not in df.columns:

        return df

    df = df[
        df["Nam_clean"].notna()
    ]

    return df.reset_index(drop=True)

# PHẦN 3 (cuối file preprocess.py)

# ==========================================================
# PREPARE DATASET FOR EDA
# ==========================================================

def prepare_eda_data(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Chuẩn bị dữ liệu phục vụ EDA.

    Bao gồm:
        - Chuẩn hóa giá
        - Chuẩn hóa năm
        - Chuẩn hóa km
        - Chuẩn hóa quận
        - Chuẩn hóa text
        - Chuẩn hóa categorical
    """

    df = df.copy()

    df = create_clean_price_column(df)

    df = create_clean_year_column(df)

    df = create_clean_km_column(df)

    df = create_clean_district_column(df)

    df = create_clean_text_columns(df)

    df = clean_categorical_columns(df)
    
    from src.feature_engineering import map_segment

    df["PhanKhucXe"] = df.apply(
        map_segment,
        axis=1,
    )

    return df


# ==========================================================
# CLEAN DATASET
# ==========================================================

def clean_dataset(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Pipeline tiền xử lý đầy đủ.

    Được dùng trước khi
        Feature Engineering
    """

    df = prepare_eda_data(df)

    df = remove_duplicate_rows(df)

    df = remove_invalid_price(df)

    df = remove_invalid_year(df)

    df = df.reset_index(drop=True)

    return df


# ==========================================================
# DATA QUALITY SUMMARY
# ==========================================================

def get_data_quality_summary(
    df: pd.DataFrame
) -> dict:
    """
    Tổng hợp chất lượng dữ liệu.
    """

    summary = {

        "rows": len(df),

        "columns": len(df.columns),

        "duplicates": int(df.duplicated().sum()),

        "missing_cells": int(df.isna().sum().sum()),

        "missing_percent": round(
            float(df.isna().sum().sum())
            /
            (len(df) * len(df.columns))
            * 100,
            2,
        ),

    }

    return summary


# ==========================================================
# MISSING REPORT
# ==========================================================

def get_missing_report(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Báo cáo Missing Value.
    """

    report = pd.DataFrame({

        "Missing": df.isna().sum(),

        "Percent": (
            df.isna().mean() * 100
        ).round(2)

    })

    report = report.sort_values(
        "Missing",
        ascending=False
    )

    return report


# ==========================================================
# NUMERIC SUMMARY
# ==========================================================

def get_numeric_summary(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Thống kê biến số.
    """

    numeric = df.select_dtypes(
        include=np.number
    )

    return numeric.describe().T


# ==========================================================
# VALIDATE BATCH FILE
# ==========================================================

def validate_batch_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    optional_columns: list[str] | None = None,
):
    """
    Kiểm tra file upload.

    Trả về

        is_valid

        message
    """

    cols = [
        c.lower().strip()
        for c in df.columns
    ]

    missing = []

    for c in required_columns:

        if c.lower() not in cols:

            missing.append(c)

    if len(missing) > 0:

        return (
            False,
            "Thiếu cột: "
            + ", ".join(missing)
        )

    return (
        True,
        "OK"
    )


# ==========================================================
# STANDARDIZE COLUMN NAME
# ==========================================================

def standardize_column_names(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Chuẩn hóa tên cột.

    bỏ khoảng trắng

    đổi space thành _
    """

    df = df.copy()

    new_cols = []

    for c in df.columns:

        c = str(c)

        c = c.strip()

        c = c.replace(" ", "_")

        new_cols.append(c)

    df.columns = new_cols

    return df


# ==========================================================
# CHECK REQUIRED COLUMNS
# ==========================================================

def has_required_columns(
    df: pd.DataFrame,
    columns: list[str]
) -> bool:

    return all(
        c in df.columns
        for c in columns
    )


# ==========================================================
# SAFE COPY
# ==========================================================

def safe_copy(
    df: pd.DataFrame
) -> pd.DataFrame:

    return df.copy(deep=True)

# ==========================================================
# FORM OPTIONS
# ==========================================================

def get_form_options(
    df: pd.DataFrame
) -> dict:
    """
    Chuẩn bị danh sách lựa chọn cho giao diện Streamlit.

    Returns
    -------
    dict
    """

    df = clean_dataset(df)

    def unique_values(col: str):

        if col not in df.columns:
            return []

        return sorted(
            df[col]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

    # -----------------------------
    # Brand
    # -----------------------------

    brands = unique_values("Thương hiệu")

    # -----------------------------
    # Model theo Brand
    # -----------------------------

    models_by_brand = {}

    if (
        "Thương hiệu" in df.columns
        and
        "Dòng xe" in df.columns
    ):

        for brand in brands:

            models = (

                df.loc[
                    df["Thương hiệu"] == brand,
                    "Dòng xe"
                ]

                .dropna()

                .astype(str)

                .str.strip()

                .unique()

                .tolist()

            )

            models_by_brand[brand] = sorted(models)

    # -----------------------------
    # District
    # -----------------------------

    districts = []

    if "Quan_clean" in df.columns:

        districts = sorted(

            df["Quan_clean"]

            .dropna()

            .astype(str)

            .unique()

            .tolist()

        )

    elif "Địa chỉ" in df.columns:

        districts = sorted(

            df["Địa chỉ"]

            .dropna()

            .apply(extract_district)

            .unique()

            .tolist()

        )

    options = {

        "brands":

            brands,

        "models":

            unique_values("Dòng xe"),

        "models_by_brand":

            models_by_brand,

        "vehicle_types":

            unique_values("Loại xe"),

        "engine_sizes":

            unique_values("Dung tích xe"),

        "origins":

            unique_values("Xuất xứ"),

        "districts":

            districts,

    }

    return options