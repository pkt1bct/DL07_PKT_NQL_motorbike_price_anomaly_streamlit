# -*- coding: utf-8 -*-
"""
src/anomaly_detector.py

Phát hiện bất thường giá xe máy cũ, port từ phần:
    "Bài toán 2: Phát hiện bất thường giá"
trong tài liệu P2_Bat_thuong_Gia.

Bốn tín hiệu của P2:
1. Residual-z theo nhóm PhanKhuc, fallback về PhanKhucXe nếu nhóm < 10 mẫu.
2. Vi phạm giới hạn P1/P99 của Giá_clean theo nhóm.
3. Nằm ngoài khoảng P10/P90 của Giá_clean theo nhóm.
4. Isolation Forest trên:
       TuoiXe, SoKm, Km_per_year, Giá_clean,
       residual, resid_z, gia_du_doan.

Mỗi tín hiệu được chuẩn hóa thành percentile rank 0-100.

Điểm tổng hợp chính thức:
    anomaly_score =
        0.35 * score_resid
      + 0.20 * score_minmax
      + 0.25 * score_range
      + 0.20 * score_iso

Ngưỡng:
    percentile 95% của anomaly_score
    => top 5% được đánh dấu bất thường.

Module tương thích với app.py:
    analyze_price_anomaly(
        input_df=input_df,
        asking_price=asking_price,
        prediction_result=prediction_result,
        reference_df=df_eda,
    )

    analyze_batch_anomalies(
        prediction_df=batch_prediction_df,
        reference_df=df_eda,
    )

Lưu ý triển khai:
- Notebook P2 tính toàn bộ tín hiệu trên df_pred, tức DataFrame đã có dự đoán
  cho toàn bộ tập dữ liệu.
- Với batch prediction, module dùng chính các dự đoán trong batch, đúng công thức P2.
- Với một xe đơn lẻ, app hiện không truyền models vào anomaly_detector. Nếu reference_df
  chưa có cột gia_du_doan/predicted_price, module tạo giá dự đoán tham chiếu bằng median
  Giá_clean theo PhanKhuc (fallback PhanKhucXe). Đây là phương án hiệu chỉnh cần thiết
  để tái tạo phân phối residual trong kiến trúc app hiện tại; công thức chấm điểm sau đó
  vẫn giữ nguyên P2.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import (
    REFERENCE_YEAR_P2,
    build_input_dataframe,
)
from src.preprocess import parse_money_to_million


# ============================================================================
# HẰNG SỐ ĐÚNG P2
# ============================================================================

RANDOM_STATE = 42
MIN_GROUP_SIZE = 10

ISO_N_ESTIMATORS = 300
ISO_CONTAMINATION = 0.05

ANOMALY_QUANTILE = 0.65

WEIGHT_RESID = 0.35
WEIGHT_MINMAX = 0.20
WEIGHT_RANGE = 0.25
WEIGHT_ISO = 0.20

EQUAL_WEIGHT = 0.25

RESID_LOW_THRESHOLD = -3.0
RESID_HIGH_THRESHOLD = 3.0

# Biên độ chênh lệch giá được xem là bình thường
NORMAL_PRICE_DEVIATION_PCT = 20.0

MAD_SCALE = 1.4826

ISO_FEATURE_COLUMNS = [
    "TuoiXe",
    "SoKm",
    "Km_per_year",
    "Giá_clean",
    "residual",
    "resid_z",
    "gia_du_doan",
]

SIGNAL_SCORE_COLUMNS = [
    "score_resid",
    "score_minmax",
    "score_range",
    "score_iso",
]


assert abs(
    WEIGHT_RESID
    + WEIGHT_MINMAX
    + WEIGHT_RANGE
    + WEIGHT_ISO
    - 1.0
) < 1e-9, "Tổng trọng số phải bằng 1."


# ============================================================================
# DATA CLASS
# ============================================================================

@dataclass(frozen=True)
class AnomalyConfiguration:
    min_group_size: int = MIN_GROUP_SIZE
    random_state: int = RANDOM_STATE
    iso_n_estimators: int = ISO_N_ESTIMATORS
    iso_contamination: float = ISO_CONTAMINATION
    anomaly_quantile: float = ANOMALY_QUANTILE
    
    weight_resid: float = WEIGHT_RESID
    weight_minmax: float = WEIGHT_MINMAX
    weight_range: float = WEIGHT_RANGE
    weight_iso: float = WEIGHT_ISO

    # Biên độ chênh lệch giá được xem là bình thường (%)
    normal_price_deviation_pct: float = NORMAL_PRICE_DEVIATION_PCT

DEFAULT_CONFIG = AnomalyConfiguration()


# ============================================================================
# CHUẨN HÓA DATAFRAME / GIÁ
# ============================================================================

def _ensure_dataframe(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        frame = data.copy(deep=True)
    elif isinstance(data, pd.Series):
        frame = data.to_frame().T.copy(deep=True)
    elif isinstance(data, Mapping):
        frame = pd.DataFrame([dict(data)])
    else:
        raise TypeError(
            "Dữ liệu phải là Mapping, pandas Series hoặc pandas DataFrame."
        )

    if frame.empty:
        raise ValueError("DataFrame không có bản ghi.")

    return frame.reset_index(drop=True)


def _coerce_price_series(series: pd.Series) -> pd.Series:
    """
    Chuẩn hóa giá về triệu đồng theo preprocess.py.
    """
    numeric = pd.to_numeric(series, errors="coerce")

    # Các giá numeric rất lớn được xem là VND.
    mask_large = numeric.abs() > 100_000
    numeric.loc[mask_large] = numeric.loc[mask_large] / 1_000_000

    missing = numeric.isna()
    if missing.any():
        numeric.loc[missing] = series.loc[missing].apply(
            parse_money_to_million
        )

    return numeric.astype(float)


def _find_price_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "Giá_clean",
        "Gia_clean",
        "gia_thuc_te",
        "Giá rao",
        "gia_rao",
        "Gia_rao_clean",
        "Giá",
        "price",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    return None


def _find_prediction_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "gia_du_doan",
        "predicted_price",
        "Giá dự đoán",
        "Gia_du_doan",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    return None


def _standardize_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    price_column = _find_price_column(result)
    if price_column is None:
        raise ValueError(
            "Không tìm thấy cột giá thực tế/giá rao."
        )

    result["Giá_clean"] = _coerce_price_series(
        result[price_column]
    )
    result["Gia_clean"] = result["Giá_clean"]
    result["gia_thuc_te"] = result["Giá_clean"]

    prediction_column = _find_prediction_column(result)
    if prediction_column is not None:
        result["gia_du_doan"] = _coerce_price_series(
            result[prediction_column]
        )

    return result


def _ensure_feature_columns(
    df: pd.DataFrame,
    reference_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Bổ sung các feature PhanKhuc, PhanKhucXe, TuoiXe, SoKm, Km_per_year.

    Nếu DataFrame đã có đủ thì giữ nguyên.
    """
    required = {
        "PhanKhuc",
        "PhanKhucXe",
        "TuoiXe",
        "SoKm",
        "Km_per_year",
    }

    if required.issubset(df.columns):
        return df.copy()

    reference = df if reference_df is None else reference_df

    try:
        featured = build_input_dataframe(
            prediction_values=df,
            reference_df=reference,
        )
    except Exception as exc:
        missing = sorted(required.difference(df.columns))
        raise ValueError(
            "Không thể tạo các feature phục vụ anomaly detection. "
            f"Thiếu: {', '.join(missing)}. Lỗi gốc: {exc}"
        ) from exc

    # Giữ các cột kết quả dự đoán/giá từ input nếu build_input_dataframe
    # không bảo toàn đầy đủ.
    for column in df.columns:
        if column not in featured.columns:
            featured[column] = df[column].to_numpy()

    return featured


# ============================================================================
# TÍN HIỆU 1: ROBUST RESIDUAL-Z
# ============================================================================

def robust_group_stats(
    data: pd.DataFrame,
    group_col: str,
    value_col: str,
    fallback_col: str,
    min_size: int = MIN_GROUP_SIZE,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Đúng P2:
    - median theo group_col
    - MAD theo group_col
    - nếu nhóm < min_size, fallback về fallback_col
    """
    _require_columns(
        data,
        [group_col, value_col, fallback_col],
    )

    group_size = (
        data.groupby(group_col, dropna=False)[value_col]
        .transform("size")
    )

    primary_median = (
        data.groupby(group_col, dropna=False)[value_col]
        .transform("median")
    )

    primary_mad = (
        data.groupby(group_col, dropna=False)[value_col]
        .transform(
            lambda series: (
                series - series.median()
            ).abs().median()
        )
    )

    fallback_median = (
        data.groupby(fallback_col, dropna=False)[value_col]
        .transform("median")
    )

    fallback_mad = (
        data.groupby(fallback_col, dropna=False)[value_col]
        .transform(
            lambda series: (
                series - series.median()
            ).abs().median()
        )
    )

    use_fallback = group_size < int(min_size)

    final_median = primary_median.where(
        ~use_fallback,
        fallback_median,
    )

    final_mad = primary_mad.where(
        ~use_fallback,
        fallback_mad,
    )

    return final_median, final_mad, use_fallback


# ============================================================================
# TÍN HIỆU 2 & 3: QUANTILE FALLBACK
# ============================================================================

def quantile_with_fallback(
    data: pd.DataFrame,
    group_col: str,
    value_col: str,
    fallback_col: str,
    q: float,
    min_size: int = MIN_GROUP_SIZE,
) -> pd.Series:
    """
    Đúng P2:
    quantile theo PhanKhuc; nhóm ít mẫu fallback về PhanKhucXe.
    """
    if not 0 <= q <= 1:
        raise ValueError("q phải thuộc [0, 1].")

    _require_columns(
        data,
        [group_col, value_col, fallback_col],
    )

    group_size = (
        data.groupby(group_col, dropna=False)[value_col]
        .transform("size")
    )

    primary_q = (
        data.groupby(group_col, dropna=False)[value_col]
        .transform(lambda series: series.quantile(q))
    )

    fallback_q = (
        data.groupby(fallback_col, dropna=False)[value_col]
        .transform(lambda series: series.quantile(q))
    )

    return primary_q.where(
        group_size >= int(min_size),
        fallback_q,
    )


def price_gap_score(row: pd.Series) -> float:
    """
    Tín hiệu 2 P2: độ lệch ngoài P1/P99.
    """
    low = row.get("GiaMin_P1")
    high = row.get("GiaMax_P99")
    actual = row.get("Giá_clean")

    if pd.isna(low) or pd.isna(high) or pd.isna(actual):
        return 0.0

    if actual < low:
        if low == 0:
            return 0.0
        return float((low - actual) / low)

    if actual > high:
        if high == 0:
            return 0.0
        return float((actual - high) / high)

    return 0.0


def outside_range_score(row: pd.Series) -> float:
    """
    Tín hiệu 3 P2: độ lệch ngoài P10/P90.
    """
    low = row.get("P10")
    high = row.get("P90")
    actual = row.get("Giá_clean")

    if (
        pd.isna(low)
        or pd.isna(high)
        or pd.isna(actual)
        or high == low
    ):
        return 0.0

    width = high - low

    if actual < low:
        return float((low - actual) / width)

    if actual > high:
        return float((actual - high) / width)

    return 0.0


# ============================================================================
# TÍN HIỆU 4: ISOLATION FOREST
# ============================================================================

def build_iso_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Đúng danh sách feature Isolation Forest của P2.
    """
    _require_columns(df, ISO_FEATURE_COLUMNS)

    return (
        df[ISO_FEATURE_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )


def fit_isolation_forest(
    iso_features: pd.DataFrame,
    config: AnomalyConfiguration = DEFAULT_CONFIG,
) -> tuple[StandardScaler, IsolationForest, np.ndarray, np.ndarray]:
    """
    Đúng P2:
        StandardScaler()
        IsolationForest(
            n_estimators=300,
            contamination=0.05,
            random_state=42,
        )
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(iso_features)

    isolation_forest = IsolationForest(
        n_estimators=config.iso_n_estimators,
        contamination=config.iso_contamination,
        random_state=config.random_state,
    )

    isolation_forest.fit(scaled)

    # score_samples thấp = bất thường => đảo dấu.
    iso_score = -isolation_forest.score_samples(scaled)

    # -1 bất thường, 1 bình thường.
    iso_labels = isolation_forest.predict(scaled)

    return scaler, isolation_forest, iso_score, iso_labels


# ============================================================================
# PERCENTILE SCORE VÀ CỜ
# ============================================================================

def to_percentile_score(series: pd.Series) -> pd.Series:
    """
    Đúng P2:
        s.rank(pct=True) * 100
    """
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    return numeric.rank(pct=True) * 100


def flag_by_threshold(
    value: Any,
    low_thresh: float,
    high_thresh: float,
) -> str:
    if pd.isna(value):
        return "Không đủ dữ liệu"

    if value <= low_thresh:
        return "Quá rẻ"

    if value >= high_thresh:
        return "Quá đắt"

    return "Bình thường"


def flag_minmax(row: pd.Series) -> str:
    low = row.get("GiaMin_P1")
    high = row.get("GiaMax_P99")
    actual = row.get("Giá_clean")

    if pd.isna(low) or pd.isna(high) or pd.isna(actual):
        return "Không đủ dữ liệu"

    if actual < low:
        return "Quá rẻ"

    if actual > high:
        return "Quá đắt"

    return "Bình thường"


def flag_range(row: pd.Series) -> str:
    low = row.get("P10")
    high = row.get("P90")
    actual = row.get("Giá_clean")

    if pd.isna(low) or pd.isna(high) or pd.isna(actual):
        return "Không đủ dữ liệu"

    if actual < low:
        return "Quá rẻ"

    if actual > high:
        return "Quá đắt"

    return "Bình thường"


def flag_final(
    row: pd.Series,
    config: AnomalyConfiguration = DEFAULT_CONFIG,
) -> str:
    """
    Kết luận cuối cùng về mức giá.

    - Lệch <= biên độ cấu hình: Bình thường
    - Lệch > biên độ nhưng score chưa vượt threshold: Bình thường
    - Score vượt threshold và giá thấp: Quá rẻ
    - Score vượt threshold và giá cao: Quá đắt
    """

    predicted = float(
        row.get("gia_du_doan", 0.0)
    )

    residual = float(
        row.get("residual", 0.0)
    )

    if predicted != 0:
        deviation_pct = residual / predicted * 100
    else:
        deviation_pct = 0.0

    # Biên giá hợp lý lấy từ cấu hình
    if abs(deviation_pct) <= config.normal_price_deviation_pct:
        return "Bình thường"

    # Điểm chưa đủ cao
    if not bool(row.get("is_anomaly", False)):
        return "Bình thường"

    # Bất thường thực sự
    if residual < 0:
        return "Quá rẻ"

    return "Quá đắt"


# ============================================================================
# PIPELINE CHẤM ĐIỂM ĐÚNG P2
# ============================================================================

def score_anomaly_dataframe(
    df_pred: pd.DataFrame,
    config: AnomalyConfiguration = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Nhận DataFrame đã có:
    - Giá_clean
    - gia_du_doan
    - PhanKhuc
    - PhanKhucXe
    - TuoiXe, SoKm, Km_per_year

    và tạo toàn bộ cột anomaly của P2.
    """
    result = _ensure_dataframe(df_pred)
    result = _standardize_price_columns(result)

    _require_columns(
        result,
        [
            "Giá_clean",
            "gia_du_doan",
            "PhanKhuc",
            "PhanKhucXe",
            "TuoiXe",
            "SoKm",
            "Km_per_year",
        ],
    )

    result["gia_du_doan"] = _coerce_price_series(
        result["gia_du_doan"]
    )

    # Notebook:
    # gia_thuc_te = Giá_clean
    # residual = gia_thuc_te - gia_du_doan
    result["gia_thuc_te"] = result["Giá_clean"]
    result["residual"] = (
        result["gia_thuc_te"]
        - result["gia_du_doan"]
    )

    # --------------------------------------------------------
    # Tín hiệu 1
    # --------------------------------------------------------
    (
        result["resid_median"],
        resid_mad,
        result["used_fallback_group"],
    ) = robust_group_stats(
        result,
        group_col="PhanKhuc",
        value_col="residual",
        fallback_col="PhanKhucXe",
        min_size=config.min_group_size,
    )

    result["resid_mad"] = resid_mad.replace(0, np.nan)

    result["resid_z"] = (
        (
            result["residual"]
            - result["resid_median"]
        )
        / (MAD_SCALE * result["resid_mad"])
    ).fillna(0)

    # --------------------------------------------------------
    # Tín hiệu 2
    # --------------------------------------------------------
    result["GiaMin_P1"] = quantile_with_fallback(
        result,
        "PhanKhuc",
        "Giá_clean",
        "PhanKhucXe",
        0.01,
        config.min_group_size,
    )

    result["GiaMax_P99"] = quantile_with_fallback(
        result,
        "PhanKhuc",
        "Giá_clean",
        "PhanKhucXe",
        0.99,
        config.min_group_size,
    )

    result["minmax_gap"] = result.apply(
        price_gap_score,
        axis=1,
    )

    # --------------------------------------------------------
    # Tín hiệu 3
    # --------------------------------------------------------
    result["P10"] = quantile_with_fallback(
        result,
        "PhanKhuc",
        "Giá_clean",
        "PhanKhucXe",
        0.10,
        config.min_group_size,
    )

    result["P90"] = quantile_with_fallback(
        result,
        "PhanKhuc",
        "Giá_clean",
        "PhanKhucXe",
        0.90,
        config.min_group_size,
    )

    result["range_gap"] = result.apply(
        outside_range_score,
        axis=1,
    )

    # --------------------------------------------------------
    # Tín hiệu 4
    # --------------------------------------------------------
    iso_features = build_iso_features(result)

    _, _, iso_score, iso_labels = fit_isolation_forest(
        iso_features,
        config=config,
    )

    result["iso_score"] = iso_score

    # --------------------------------------------------------
    # Tín hiệu 5
    # --------------------------------------------------------
    result["score_resid"] = to_percentile_score(
        result["resid_z"].abs()
    )

    result["score_minmax"] = to_percentile_score(
        result["minmax_gap"]
    )

    result["score_range"] = to_percentile_score(
        result["range_gap"]
    )

    result["score_iso"] = to_percentile_score(
        result["iso_score"]
    )

    # Baseline trọng số đều.
    result["anomaly_score_equal"] = EQUAL_WEIGHT * (
        result["score_resid"]
        + result["score_minmax"]
        + result["score_range"]
        + result["score_iso"]
    )

    threshold_equal = float(
        result["anomaly_score_equal"]
        .quantile(config.anomaly_quantile)
    )

    result["threshold_equal"] = threshold_equal

    result["is_anomaly_equal"] = (
        result["anomaly_score_equal"]
        >= threshold_equal
    )

    # Trọng số chính thức P2.
    result["anomaly_score"] = (
        config.weight_resid * result["score_resid"]
        + config.weight_minmax * result["score_minmax"]
        + config.weight_range * result["score_range"]
        + config.weight_iso * result["score_iso"]
    )

    threshold = float(
        result["anomaly_score"]
        .quantile(config.anomaly_quantile)
    )

    result["threshold"] = threshold

    # --------------------------------------------------------
    # Độ lệch giữa giá rao và giá mô hình (%)
    # --------------------------------------------------------
    result["deviation_pct"] = np.where(
        result["gia_du_doan"] != 0,
        result["residual"] / result["gia_du_doan"] * 100,
        0.0,
    )

    # Cờ theo anomaly score thuần túy
    result["score_exceeds_threshold"] = (
        result["anomaly_score"] >= threshold
    )

    # --------------------------------------------------------
    # CỜ BẤT THƯỜNG CUỐI CÙNG
    #
    # Một mức giá chỉ được xem là bất thường khi:
    # 1. anomaly score vượt threshold
    # 2. giá rao lệch hơn biên độ cho phép so với giá dự đoán
    # --------------------------------------------------------
    result["is_anomaly"] = (
        result["score_exceeds_threshold"]
        & (
            result["deviation_pct"].abs()
            > config.normal_price_deviation_pct
        )
    )

    # --------------------------------------------------------
    # Các cờ song song
    # --------------------------------------------------------
    result["flag_resid"] = result["resid_z"].apply(
        lambda value: flag_by_threshold(
            value,
            RESID_LOW_THRESHOLD,
            RESID_HIGH_THRESHOLD,
        )
    )

    result["flag_minmax"] = result.apply(
        flag_minmax,
        axis=1,
    )

    result["flag_range"] = result.apply(
        flag_range,
        axis=1,
    )

    result["flag_iso"] = np.where(
        iso_labels == -1,
        "Bất thường",
        "Bình thường",
    )

    result["flag_final"] = result.apply(
        lambda row: flag_final(row, config=config),
        axis=1,
    )

    result["label"] = result["flag_final"]

    return result.reset_index(drop=True)


# ============================================================================
# CHUẨN BỊ REFERENCE CHO SINGLE ANALYSIS
# ============================================================================

def _prepare_reference_dataframe(
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Chuẩn bị reference_df thành dạng feature + Giá_clean.

    Nếu reference đã có dự đoán thì dùng trực tiếp.
    Nếu chưa có, tạo benchmark median theo nhóm để sinh residual tham chiếu.
    """
    reference = _ensure_dataframe(reference_df)
    reference = _standardize_price_columns(reference)

    reference = _ensure_feature_columns(
        reference,
        reference_df=reference,
    )

    prediction_column = _find_prediction_column(reference)

    if prediction_column is not None:
        reference["gia_du_doan"] = _coerce_price_series(
            reference[prediction_column]
        )
        return reference

    # Fallback triển khai: median giá theo PhanKhuc,
    # nhóm < 10 dùng median PhanKhucXe.
    group_size = (
        reference.groupby("PhanKhuc", dropna=False)["Giá_clean"]
        .transform("size")
    )

    primary_median = (
        reference.groupby("PhanKhuc", dropna=False)["Giá_clean"]
        .transform("median")
    )

    fallback_median = (
        reference.groupby("PhanKhucXe", dropna=False)["Giá_clean"]
        .transform("median")
    )

    reference["gia_du_doan"] = primary_median.where(
        group_size >= MIN_GROUP_SIZE,
        fallback_median,
    )

    global_median = reference["Giá_clean"].median()

    reference["gia_du_doan"] = (
        reference["gia_du_doan"]
        .fillna(global_median)
    )

    return reference


def _prediction_price_from_result(
    prediction_result: Mapping[str, Any],
) -> float:
    if "predicted_price" not in prediction_result:
        raise ValueError(
            "prediction_result thiếu predicted_price."
        )

    value = float(prediction_result["predicted_price"])

    if abs(value) > 100_000:
        value /= 1_000_000

    return value


# ============================================================================
# API CHO APP.PY: MỘT XE
# ============================================================================

def analyze_price_anomaly(
    input_df: pd.DataFrame,
    asking_price: float,
    prediction_result: Mapping[str, Any],
    reference_df: pd.DataFrame,
    config: AnomalyConfiguration = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """
    Phân tích bất thường cho một xe theo contract của app.py.

    Candidate được nối vào reference rồi chấm điểm cùng toàn bộ phân phối,
    sau đó trả về đúng dòng candidate.
    """
    candidate = _ensure_dataframe(input_df)

    if len(candidate) != 1:
        raise ValueError(
            "analyze_price_anomaly() chỉ nhận đúng một dòng."
        )

    asking_price_million = float(asking_price)

    if abs(asking_price_million) > 100_000:
        asking_price_million /= 1_000_000

    if not np.isfinite(asking_price_million):
        raise ValueError("asking_price không hợp lệ.")

    if asking_price_million <= 0:
        raise ValueError("asking_price phải lớn hơn 0.")

    predicted_price = _prediction_price_from_result(
        prediction_result
    )

    reference = _prepare_reference_dataframe(
        reference_df
    )

    candidate = _ensure_feature_columns(
        candidate,
        reference_df=reference_df,
    )

    candidate["Giá_clean"] = asking_price_million
    candidate["Gia_clean"] = asking_price_million
    candidate["gia_thuc_te"] = asking_price_million
    candidate["gia_du_doan"] = predicted_price

    candidate_marker = "__candidate_for_single_analysis__"
    reference_marker = "__reference_for_single_analysis__"

    reference["_anomaly_row_marker"] = reference_marker
    candidate["_anomaly_row_marker"] = candidate_marker

    combined = pd.concat(
        [reference, candidate],
        ignore_index=True,
        sort=False,
    )

    scored = score_anomaly_dataframe(
        combined,
        config=config,
    )

    candidate_rows = scored[
        scored["_anomaly_row_marker"] == candidate_marker
    ]

    if len(candidate_rows) != 1:
        raise RuntimeError(
            "Không xác định được dòng candidate sau khi chấm điểm."
        )

    row = candidate_rows.iloc[0]

    components = {
        "Residual-z": float(row["score_resid"]),
        "P1/P99": float(row["score_minmax"]),
        "P10/P90": float(row["score_range"]),
        "Isolation Forest": float(row["score_iso"]),
    }

    residual = float(row["residual"])
    predicted = float(row["gia_du_doan"])
    deviation_pct = (
        residual / predicted * 100
        if predicted != 0
        else 0.0
    )

    return {
        "predicted_price": predicted,
        "asking_price": float(row["Giá_clean"]),
        "residual": residual,
        "deviation_pct": deviation_pct,
        "resid_z": float(row["resid_z"]),
        "resid_median": float(row["resid_median"])
        if pd.notna(row["resid_median"])
        else np.nan,
        "resid_mad": float(row["resid_mad"])
        if pd.notna(row["resid_mad"])
        else np.nan,
        "used_fallback_group": bool(row["used_fallback_group"]),
        "GiaMin_P1": float(row["GiaMin_P1"])
        if pd.notna(row["GiaMin_P1"])
        else np.nan,
        "GiaMax_P99": float(row["GiaMax_P99"])
        if pd.notna(row["GiaMax_P99"])
        else np.nan,
        "P10": float(row["P10"])
        if pd.notna(row["P10"])
        else np.nan,
        "P90": float(row["P90"])
        if pd.notna(row["P90"])
        else np.nan,
        "minmax_gap": float(row["minmax_gap"]),
        "range_gap": float(row["range_gap"]),
        "iso_score": float(row["iso_score"]),
        "score_resid": float(row["score_resid"]),
        "score_minmax": float(row["score_minmax"]),
        "score_range": float(row["score_range"]),
        "score_iso": float(row["score_iso"]),
        "anomaly_score_equal": float(row["anomaly_score_equal"]),
        "threshold_equal": float(row["threshold_equal"]),
        "is_anomaly_equal": bool(row["is_anomaly_equal"]),
        "anomaly_score": float(row["anomaly_score"]),
        "threshold": float(row["threshold"]),
        
        "anomaly_quantile": float(
            config.anomaly_quantile
        ),
        
        # Biên độ giá bình thường thực tế được sử dụng
        "normal_price_deviation_pct": float(
            config.normal_price_deviation_pct
        ),
        
        "is_anomaly": bool(row["is_anomaly"]),
        "flag_resid": str(row["flag_resid"]),
        "flag_minmax": str(row["flag_minmax"]),
        "flag_range": str(row["flag_range"]),
        "flag_iso": str(row["flag_iso"]),
        "flag_final": str(row["flag_final"]),
        "label": str(row["flag_final"]),
        "components": components,
        "segment": prediction_result.get(
            "segment",
            row.get("PhanKhucXe", "Không xác định"),
        ),
        "model_segment": prediction_result.get(
            "model_segment",
            "",
        ),
        "model_name": prediction_result.get(
            "model_name",
            "",
        ),
    }


# ============================================================================
# API CHO APP.PY: BATCH
# ============================================================================

def analyze_batch_anomalies(
    prediction_df: pd.DataFrame,
    reference_df: pd.DataFrame | None = None,
    config: AnomalyConfiguration = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Phân tích bất thường hàng loạt.

    prediction_df từ predictor.predict_batch() đã có:
    - predicted_price
    - gia_du_doan
    - PhanKhuc
    - PhanKhucXe
    - feature P2

    Giá thực tế ưu tiên:
    - Gia_rao_clean
    - Giá rao / gia_rao
    - Giá_clean / Gia_clean
    """
    batch = _ensure_dataframe(prediction_df)

    # Đảm bảo feature.
    batch = _ensure_feature_columns(
        batch,
        reference_df=reference_df if reference_df is not None else batch,
    )

    # predicted_price -> gia_du_doan.
    prediction_column = _find_prediction_column(batch)
    if prediction_column is None:
        raise ValueError(
            "prediction_df thiếu predicted_price/gia_du_doan."
        )

    batch["gia_du_doan"] = _coerce_price_series(
        batch[prediction_column]
    )

    # Giá rao thực tế.
    price_candidates = [
        "Gia_rao_clean",
        "Giá rao",
        "gia_rao",
        "Giá_clean",
        "Gia_clean",
        "Giá",
    ]

    actual_column = None
    for column in price_candidates:
        if column in batch.columns:
            values = _coerce_price_series(batch[column])
            if values.notna().any():
                actual_column = column
                batch["Giá_clean"] = values
                break

    if actual_column is None:
        raise ValueError(
            "prediction_df thiếu giá rao thực tế để phát hiện bất thường."
        )

    batch["Gia_clean"] = batch["Giá_clean"]
    batch["gia_thuc_te"] = batch["Giá_clean"]

    scored = score_anomaly_dataframe(
        batch,
        config=config,
    )

    # Các alias tiện dụng cho app/recommendation/export.
    scored["predicted_price"] = scored["gia_du_doan"]
    scored["asking_price"] = scored["Giá_clean"]
    scored["label"] = scored["flag_final"]

    scored["deviation_pct"] = np.where(
        scored["gia_du_doan"] != 0,
        scored["residual"] / scored["gia_du_doan"] * 100,
        0.0,
    )

    return scored.reset_index(drop=True)


# ============================================================================
# BÁO CÁO / TIỆN ÍCH
# ============================================================================

def get_anomaly_summary(
    scored_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Tổng hợp các chỉ số notebook P2 để hiển thị/report.
    """
    frame = _ensure_dataframe(scored_df)

    _require_columns(
        frame,
        [
            "is_anomaly",
            "is_anomaly_equal",
            "anomaly_score",
            "anomaly_score_equal",
            "threshold",
            "threshold_equal",
        ],
    )

    anomaly_count = int(frame["is_anomaly"].sum())
    equal_count = int(frame["is_anomaly_equal"].sum())

    overlap = int(
        (
            frame["is_anomaly"]
            & frame["is_anomaly_equal"]
        ).sum()
    )

    overlap_pct = (
        overlap / anomaly_count
        if anomaly_count
        else 0.0
    )

    return {
        "rows": int(len(frame)),
        "anomaly_count": anomaly_count,
        "anomaly_rate": float(
            anomaly_count / len(frame)
            if len(frame)
            else 0.0
        ),
        "equal_anomaly_count": equal_count,
        "threshold": float(frame["threshold"].iloc[0]),
        "threshold_equal": float(
            frame["threshold_equal"].iloc[0]
        ),
        "overlap": overlap,
        "overlap_pct": overlap_pct,
        "fallback_count": int(
            frame.get(
                "used_fallback_group",
                pd.Series(False, index=frame.index),
            ).sum()
        ),
        "flag_final_counts": (
            frame.get(
                "flag_final",
                pd.Series(dtype=str),
            )
            .value_counts()
            .to_dict()
        ),
    }


def get_signal_correlation(
    scored_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ma trận tương quan 4 tín hiệu như notebook P2.
    """
    frame = _ensure_dataframe(scored_df)
    _require_columns(frame, SIGNAL_SCORE_COLUMNS)

    return frame[SIGNAL_SCORE_COLUMNS].corr()


def get_top_cheap_anomalies(
    scored_df: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    frame = _ensure_dataframe(scored_df)

    mask = (
        frame["is_anomaly"].astype(bool)
        & (frame["residual"] < 0)
    )

    return (
        frame.loc[mask]
        .sort_values("anomaly_score", ascending=False)
        .head(int(n))
        .copy()
    )


def get_top_expensive_anomalies(
    scored_df: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    frame = _ensure_dataframe(scored_df)

    mask = (
        frame["is_anomaly"].astype(bool)
        & (frame["residual"] > 0)
    )

    return (
        frame.loc[mask]
        .sort_values("anomaly_score", ascending=False)
        .head(int(n))
        .copy()
    )


# ============================================================================
# VALIDATION
# ============================================================================

def _require_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> None:
    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Thiếu các cột bắt buộc: "
            + ", ".join(missing)
        )


__all__ = [
    "AnomalyConfiguration",
    "DEFAULT_CONFIG",
    "MIN_GROUP_SIZE",
    "robust_group_stats",
    "quantile_with_fallback",
    "price_gap_score",
    "outside_range_score",
    "build_iso_features",
    "fit_isolation_forest",
    "to_percentile_score",
    "flag_by_threshold",
    "flag_minmax",
    "flag_range",
    "flag_final",
    "score_anomaly_dataframe",
    "analyze_price_anomaly",
    "analyze_batch_anomalies",
    "get_anomaly_summary",
    "get_signal_correlation",
    "get_top_cheap_anomalies",
    "get_top_expensive_anomalies",
]
