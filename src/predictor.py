# -*- coding: utf-8 -*-
"""
src/predictor.py

Module dự đoán giá được port từ notebook P2.ipynb và tích hợp với:
- app.py
- preprocess.py
- feature_engineering.py
- ba pipeline đã lưu:
    models/model_phothong.pkl
    models/model_trungcap.pkl
    models/model_caocap.pkl

Logic cốt lõi giữ đúng notebook P2:
1. Mỗi model là một sklearn Pipeline:
       Pipeline([("prep", preprocessor), ("model", XGBRegressor(...))])
2. Target huấn luyện là:
       log_gia = np.log1p(Giá_clean)
3. Kết quả predict của pipeline ở thang log được đổi về giá thật bằng:
       np.expm1(predicted_log_price)
4. Model được chọn theo cột PhanKhucXe:
       "phổ thông" -> model_phothong.pkl
       "trung cấp" -> model_trungcap.pkl
       "cao cấp"   -> model_caocap.pkl

Lưu ý về các phân khúc không có model riêng trong notebook:
- "siêu cao cấp", "moto cao cấp", "moto siêu sang" không được notebook P2
  fit thành model riêng.
- Khi triển khai, các nhóm này được định tuyến về model "cao cấp",
  là lựa chọn gần nhất trong ba model đã lưu.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import __main__
import pickle

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.feature_engineering import (
    MODEL_INPUT_FEATURES_P2,
    build_input_dataframe,
    select_model_input_features,
    validate_p2_features,
)


# ============================================================================
# HẰNG SỐ
# ============================================================================

SEGMENT_PHOTHONG = "phổ thông"
SEGMENT_TRUNGCAP = "trung cấp"
SEGMENT_CAOCAP = "cao cấp"

SUPPORTED_MODEL_SEGMENTS = (
    SEGMENT_PHOTHONG,
    SEGMENT_TRUNGCAP,
    SEGMENT_CAOCAP,
)

MODEL_FILE_NAMES = {
    SEGMENT_PHOTHONG: "model_phothong.pkl",
    SEGMENT_TRUNGCAP: "model_trungcap.pkl",
    SEGMENT_CAOCAP: "model_caocap.pkl",
}

# Các phân khúc được tạo bởi map_segment() trong notebook nhưng không có model riêng.
SEGMENT_FALLBACKS = {
    "siêu cao cấp": SEGMENT_CAOCAP,
    "moto cao cấp": SEGMENT_CAOCAP,
    "moto siêu sang": SEGMENT_CAOCAP,
}

DEFAULT_SEGMENT = SEGMENT_TRUNGCAP


# ============================================================================
# CLASS CẦN CHO PICKLE CỦA NOTEBOOK P2
# ============================================================================

class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    """
    Mã hóa một cột phân loại bằng trung bình target có smoothing.

    Class này giữ nguyên thuật toán được định nghĩa tại phần modeling của
    notebook P2. Nó phải tồn tại khi pickle.load() khôi phục các pipeline,
    vì preprocessor đã lưu chứa instance của class này.
    """

    def __init__(self, smoothing: float = 10):
        self.smoothing = smoothing

    def fit(self, X: Any, y: Any):
        frame = _ensure_single_column_frame(X)
        column = frame.iloc[:, 0]

        target = pd.Series(
            np.asarray(y),
            index=column.index,
        )

        self.global_mean_ = float(target.mean())

        statistics = target.groupby(column).agg(["mean", "count"])

        self.encoding_ = (
            (
                statistics["mean"] * statistics["count"]
                + self.global_mean_ * self.smoothing
            )
            / (statistics["count"] + self.smoothing)
        )

        return self

    def transform(self, X: Any) -> np.ndarray:
        _check_encoder_is_fitted(self)

        frame = _ensure_single_column_frame(X)
        column = frame.iloc[:, 0]

        encoded = (
            column
            .map(self.encoding_)
            .fillna(self.global_mean_)
        )

        return encoded.to_numpy().reshape(-1, 1)

    def get_feature_names_out(
        self,
        input_features: Any = None,
    ) -> np.ndarray:
        """
        Hỗ trợ các phiên bản sklearn mới khi truy vấn feature names.
        Không ảnh hưởng thuật toán của notebook.
        """
        if input_features is None:
            return np.asarray(["Dòng xe_target_mean"], dtype=object)

        features = list(input_features)
        base_name = str(features[0]) if features else "Dòng xe"

        return np.asarray(
            [f"{base_name}_target_mean"],
            dtype=object,
        )


def _ensure_single_column_frame(X: Any) -> pd.DataFrame:
    """Chuẩn hóa đầu vào encoder thành DataFrame đúng một cột."""
    if isinstance(X, pd.DataFrame):
        frame = X.copy()
    elif isinstance(X, pd.Series):
        frame = X.to_frame()
    else:
        array = np.asarray(X)

        if array.ndim == 1:
            array = array.reshape(-1, 1)

        frame = pd.DataFrame(array)

    if frame.shape[1] != 1:
        raise ValueError(
            "TargetMeanEncoder của notebook P2 chỉ nhận đúng một cột."
        )

    return frame


def _check_encoder_is_fitted(encoder: TargetMeanEncoder) -> None:
    if not hasattr(encoder, "global_mean_"):
        raise RuntimeError(
            "TargetMeanEncoder chưa được fit: thiếu global_mean_."
        )

    if not hasattr(encoder, "encoding_"):
        raise RuntimeError(
            "TargetMeanEncoder chưa được fit: thiếu encoding_."
        )


def _register_pickle_compatibility() -> None:
    """
    Đăng ký TargetMeanEncoder vào namespace __main__ để tương thích
    với mô hình được lưu từ notebook P2.
    """
    setattr(
        __main__,
        "TargetMeanEncoder",
        TargetMeanEncoder,
    )

class P2ModelUnpickler(pickle.Unpickler):
    """
    Khôi phục pipeline được lưu từ notebook P2.

    TargetMeanEncoder trong file pickle được ánh xạ về đúng lớp
    TargetMeanEncoder đang định nghĩa trong src.predictor.
    """

    def find_class(
        self,
        module: str,
        name: str,
    ) -> Any:
        if name == "TargetMeanEncoder":
            return TargetMeanEncoder

        return super().find_class(module, name)

# ============================================================================
# HÀM LOAD MODEL
# ============================================================================

def _load_pickle_model(
    model_path: str | Path,
) -> Any:
    """
    Đọc một pipeline .pkl và kiểm tra phương thức predict().
    """
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Đường dẫn model không phải tập tin: {path}"
        )

    _register_pickle_compatibility()

    try:
        with path.open("rb") as file:
            model = P2ModelUnpickler(file).load()

    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Không thể tải `{path.name}` vì thiếu thư viện "
            f"`{exc.name}`. Hãy cài đúng phiên bản scikit-learn, "
            "xgboost và các thư viện đã dùng khi huấn luyện "
            "notebook P2."
        ) from exc

    except AttributeError as exc:
        raise AttributeError(
            f"Không thể khôi phục class trong `{path.name}`. "
            f"Lỗi gốc: {exc}"
        ) from exc

    except Exception as exc:
        raise RuntimeError(
            f"Không thể tải model `{path}`. "
            f"Loại lỗi: {type(exc).__name__}. "
            f"Chi tiết: {exc}"
        ) from exc

    if not hasattr(model, "predict"):
        raise TypeError(
            f"Đối tượng trong `{path.name}` "
            "không có phương thức predict()."
        )

    return model


def load_models(
    phothong_path: str | Path,
    trungcap_path: str | Path,
    caocap_path: str | Path,
) -> dict[str, Any]:
    """
    Tải ba pipeline đúng với cách app.py đang gọi.

    Returns
    -------
    dict
        {
            "phổ thông": pipeline,
            "trung cấp": pipeline,
            "cao cấp": pipeline,
        }
    """
    paths = {
        SEGMENT_PHOTHONG: phothong_path,
        SEGMENT_TRUNGCAP: trungcap_path,
        SEGMENT_CAOCAP: caocap_path,
    }

    models: dict[str, Any] = {}

    for segment, path in paths.items():
        models[segment] = _load_pickle_model(path)

    return models


def load_models_from_directory(
    models_dir: str | Path,
) -> dict[str, Any]:
    """
    Hàm tiện ích tải ba model theo tên file notebook từ một thư mục.
    """
    directory = Path(models_dir)

    return load_models(
        phothong_path=directory / MODEL_FILE_NAMES[SEGMENT_PHOTHONG],
        trungcap_path=directory / MODEL_FILE_NAMES[SEGMENT_TRUNGCAP],
        caocap_path=directory / MODEL_FILE_NAMES[SEGMENT_CAOCAP],
    )


# ============================================================================
# CHUẨN HÓA PHÂN KHÚC VÀ MODEL
# ============================================================================

def normalize_segment(segment: Any) -> str:
    """
    Chuẩn hóa nhãn PhanKhucXe để định tuyến ba model của notebook P2.
    """
    if pd.isna(segment):
        return DEFAULT_SEGMENT

    normalized = str(segment).strip().lower()
    normalized = " ".join(normalized.split())

    aliases = {
        "pho thong": SEGMENT_PHOTHONG,
        "phothong": SEGMENT_PHOTHONG,
        "phổ thông": SEGMENT_PHOTHONG,

        "trung cap": SEGMENT_TRUNGCAP,
        "trungcap": SEGMENT_TRUNGCAP,
        "trung cấp": SEGMENT_TRUNGCAP,

        "cao cap": SEGMENT_CAOCAP,
        "caocap": SEGMENT_CAOCAP,
        "cao cấp": SEGMENT_CAOCAP,

        "sieu cao cap": "siêu cao cấp",
        "siêu cao cấp": "siêu cao cấp",

        "moto cao cap": "moto cao cấp",
        "moto cao cấp": "moto cao cấp",

        "moto sieu sang": "moto siêu sang",
        "moto siêu sang": "moto siêu sang",
    }

    return aliases.get(normalized, normalized)


def resolve_model_segment(segment: Any) -> str:
    """
    Trả về một trong ba segment có model đã fit.
    """
    normalized = normalize_segment(segment)

    if normalized in SUPPORTED_MODEL_SEGMENTS:
        return normalized

    if normalized in SEGMENT_FALLBACKS:
        return SEGMENT_FALLBACKS[normalized]

    return DEFAULT_SEGMENT


def get_model_for_segment(
    models: Mapping[str, Any],
    segment: Any,
) -> tuple[str, Any]:
    """
    Chọn model từ dictionary, có hỗ trợ cả key không dấu/tên file cũ.
    """
    resolved_segment = resolve_model_segment(segment)

    key_candidates = {
        SEGMENT_PHOTHONG: [
            "phổ thông",
            "pho thong",
            "phothong",
            "model_phothong",
            "model_phothong.pkl",
        ],
        SEGMENT_TRUNGCAP: [
            "trung cấp",
            "trung cap",
            "trungcap",
            "model_trungcap",
            "model_trungcap.pkl",
        ],
        SEGMENT_CAOCAP: [
            "cao cấp",
            "cao cap",
            "caocap",
            "model_caocap",
            "model_caocap.pkl",
        ],
    }

    # Ưu tiên key đúng chuẩn.
    if resolved_segment in models:
        return resolved_segment, models[resolved_segment]

    normalized_model_keys = {
        str(key).strip().lower(): key
        for key in models.keys()
    }

    for candidate in key_candidates[resolved_segment]:
        lookup = candidate.lower()

        if lookup in normalized_model_keys:
            original_key = normalized_model_keys[lookup]
            return resolved_segment, models[original_key]

    raise KeyError(
        f"Không tìm thấy model cho phân khúc `{resolved_segment}`. "
        f"Các key hiện có: {list(models.keys())}"
    )


# ============================================================================
# KIỂM TRA DỮ LIỆU ĐẦU VÀO
# ============================================================================

def _ensure_dataframe(input_df: Any) -> pd.DataFrame:
    if isinstance(input_df, pd.DataFrame):
        result = input_df.copy(deep=True)
    elif isinstance(input_df, pd.Series):
        result = input_df.to_frame().T.copy(deep=True)
    elif isinstance(input_df, Mapping):
        result = pd.DataFrame([dict(input_df)])
    else:
        raise TypeError(
            "input_df phải là Mapping, pandas Series hoặc pandas DataFrame."
        )

    if result.empty:
        raise ValueError(
            "Không có bản ghi để dự đoán."
        )

    return result.reset_index(drop=True)


def _validate_model_input(input_df: pd.DataFrame) -> None:
    missing = validate_p2_features(input_df)

    if missing:
        raise ValueError(
            "DataFrame chưa đủ feature của notebook P2. Thiếu: "
            + ", ".join(missing)
        )

    if "PhanKhucXe" not in input_df.columns:
        raise ValueError(
            "Thiếu cột PhanKhucXe để chọn model."
        )

    # Notebook dropna toàn bộ feature trước khi fit.
    # Khi predict cũng không nên để NaN ở những cột numeric pipeline passthrough.
    numeric_columns = [
        column
        for column in MODEL_INPUT_FEATURES_P2
        if column in input_df.columns
        and pd.api.types.is_numeric_dtype(input_df[column])
    ]

    columns_with_missing = [
        column
        for column in numeric_columns
        if input_df[column].isna().any()
    ]

    if columns_with_missing:
        raise ValueError(
            "Các numeric feature vẫn còn NaN trước khi predict: "
            + ", ".join(columns_with_missing)
        )


def _model_input_frame(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Lấy đúng X = numeric_features + categorical_features + ['Dòng xe']
    như notebook P2.
    """
    return select_model_input_features(
        input_df,
        strict=True,
    )


# ============================================================================
# DỰ ĐOÁN CỐT LÕI
# ============================================================================

def predict_log_price(
    model: Any,
    model_input_df: pd.DataFrame,
) -> np.ndarray:
    """
    Pipeline P2 trả về dự đoán log_gia.
    """
    try:
        predictions = model.predict(model_input_df)
    except Exception as exc:
        raise RuntimeError(
            "Pipeline không thể dự đoán. Hãy kiểm tra phiên bản thư viện, "
            "tên cột và kiểu dữ liệu có khớp lúc huấn luyện P2 hay không. "
            f"Lỗi gốc: {exc}"
        ) from exc

    predictions = np.asarray(predictions, dtype=float).reshape(-1)

    if predictions.size != len(model_input_df):
        raise RuntimeError(
            "Số lượng kết quả predict không khớp số dòng đầu vào."
        )

    if not np.isfinite(predictions).all():
        raise RuntimeError(
            "Model trả về log-price không hữu hạn."
        )

    return predictions


def inverse_log_price(log_predictions: Any) -> np.ndarray:
    """
    Đúng notebook P2:
        gia_du_doan = np.expm1(gia_du_doan_log)
    """
    log_values = np.asarray(log_predictions, dtype=float)
    prices = np.expm1(log_values)

    # Giá âm có thể xuất hiện về mặt toán học nếu log dự đoán < 0.
    # Asking price vật lý không âm, nên chặn ở 0 khi triển khai.
    return np.maximum(prices, 0.0)


def _predict_group(
    group_df: pd.DataFrame,
    model: Any,
) -> tuple[np.ndarray, np.ndarray]:
    X = _model_input_frame(group_df)
    predicted_log = predict_log_price(model, X)
    predicted_price = inverse_log_price(predicted_log)

    return predicted_log, predicted_price


def predict_price(
    input_df: pd.DataFrame,
    models: Mapping[str, Any],
    reference_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Dự đoán một xe theo contract của app.py.

    input_df phải là kết quả của:
        build_input_dataframe(..., reference_df=df_raw)

    reference_df được giữ trong signature để tương thích app.py.
    Feature engineering đã dùng reference_df trước khi gọi hàm này.

    Returns
    -------
    dict
        predicted_price:
            Giá dự đoán sau np.expm1, cùng đơn vị Giá_clean lúc train.
        predicted_log_price:
            Output trực tiếp của pipeline.
        segment:
            PhanKhucXe gốc được nhận diện.
        model_segment:
            Một trong ba model thực sự được dùng.
        model_name:
            Tên file model tương ứng.
        lower_bound, upper_bound:
            Khoảng hiển thị tham khảo ±10%; notebook không huấn luyện
            interval model riêng nên đây không phải confidence interval thống kê.
    """
    frame = _ensure_dataframe(input_df)

    if len(frame) != 1:
        raise ValueError(
            "predict_price() chỉ nhận đúng một dòng. "
            "Dùng predict_batch() cho nhiều dòng."
        )

    _validate_model_input(frame)

    original_segment = normalize_segment(
        frame.iloc[0]["PhanKhucXe"]
    )

    model_segment, model = get_model_for_segment(
        models,
        original_segment,
    )

    predicted_log, predicted_price = _predict_group(
        frame,
        model,
    )

    price = float(predicted_price[0])
    log_price = float(predicted_log[0])

    return {
        "predicted_price": price,
        "predicted_log_price": log_price,
        "segment": original_segment,
        "model_segment": model_segment,
        "model_name": MODEL_FILE_NAMES[model_segment],
        "lower_bound": max(price * 0.90, 0.0),
        "upper_bound": max(price * 1.10, 0.0),
        "input_index": frame.index[0],
    }


# ============================================================================
# BATCH PREDICTION
# ============================================================================

def _build_batch_features(
    batch_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Chuyển batch raw từ file upload thành feature DataFrame P2.
    """
    return build_input_dataframe(
        prediction_values=batch_df,
        reference_df=reference_df,
    )


def predict_batch(
    batch_df: pd.DataFrame,
    models: Mapping[str, Any],
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Dự đoán hàng loạt theo contract của app.py.

    Quy trình:
    1. build_input_dataframe() cho toàn bộ batch.
    2. Chia theo model_segment.
    3. Pipeline.predict() trên log target.
    4. np.expm1() về giá thật.
    5. Ghép kết quả theo đúng thứ tự ban đầu.

    Output giữ toàn bộ cột raw/feature và bổ sung:
    - predicted_log_price
    - predicted_price
    - gia_du_doan       (alias đúng tên notebook)
    - segment
    - model_segment
    - model_name
    - lower_bound
    - upper_bound
    """
    raw_batch = _ensure_dataframe(batch_df)

    if reference_df is None:
        raise ValueError(
            "reference_df là bắt buộc khi dự đoán batch."
        )

    featured = _build_batch_features(
        raw_batch,
        reference_df=reference_df,
    )

    _validate_model_input(featured)

    result = featured.copy()

    result["segment"] = (
        result["PhanKhucXe"]
        .apply(normalize_segment)
    )

    result["model_segment"] = (
        result["segment"]
        .apply(resolve_model_segment)
    )

    result["predicted_log_price"] = np.nan
    result["predicted_price"] = np.nan
    result["gia_du_doan"] = np.nan
    result["model_name"] = ""

    for model_segment in SUPPORTED_MODEL_SEGMENTS:
        mask = result["model_segment"] == model_segment

        if not mask.any():
            continue

        _, model = get_model_for_segment(
            models,
            model_segment,
        )

        group_df = result.loc[mask]
        predicted_log, predicted_price = _predict_group(
            group_df,
            model,
        )

        result.loc[mask, "predicted_log_price"] = predicted_log
        result.loc[mask, "predicted_price"] = predicted_price
        result.loc[mask, "gia_du_doan"] = predicted_price
        result.loc[mask, "model_name"] = MODEL_FILE_NAMES[model_segment]

    if result["predicted_price"].isna().any():
        failed_indices = result.index[
            result["predicted_price"].isna()
        ].tolist()

        raise RuntimeError(
            "Một số dòng chưa được dự đoán: "
            + ", ".join(map(str, failed_indices[:20]))
        )

    result["lower_bound"] = (
        result["predicted_price"] * 0.90
    ).clip(lower=0)

    result["upper_bound"] = (
        result["predicted_price"] * 1.10
    ).clip(lower=0)

    return result.reset_index(drop=True)


# ============================================================================
# HÀM TIỆN ÍCH
# ============================================================================

def predict_prepared_dataframe(
    input_df: pd.DataFrame,
    models: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Dự đoán một DataFrame đã feature-engineer sẵn mà không build lại feature.
    Hữu ích khi module khác đã gọi build_input_dataframe().
    """
    frame = _ensure_dataframe(input_df)
    _validate_model_input(frame)

    result = frame.copy()
    result["segment"] = result["PhanKhucXe"].apply(normalize_segment)
    result["model_segment"] = result["segment"].apply(resolve_model_segment)
    result["predicted_log_price"] = np.nan
    result["predicted_price"] = np.nan
    result["gia_du_doan"] = np.nan
    result["model_name"] = ""

    for model_segment in SUPPORTED_MODEL_SEGMENTS:
        mask = result["model_segment"] == model_segment

        if not mask.any():
            continue

        _, model = get_model_for_segment(models, model_segment)
        predicted_log, predicted_price = _predict_group(
            result.loc[mask],
            model,
        )

        result.loc[mask, "predicted_log_price"] = predicted_log
        result.loc[mask, "predicted_price"] = predicted_price
        result.loc[mask, "gia_du_doan"] = predicted_price
        result.loc[mask, "model_name"] = MODEL_FILE_NAMES[model_segment]

    return result.reset_index(drop=True)


def get_model_summary(models: Mapping[str, Any]) -> pd.DataFrame:
    """
    Tóm tắt ba model đã load để debug trên Streamlit hoặc CLI.
    """
    rows: list[dict[str, Any]] = []

    for segment in SUPPORTED_MODEL_SEGMENTS:
        try:
            _, model = get_model_for_segment(models, segment)
            rows.append(
                {
                    "segment": segment,
                    "model_file": MODEL_FILE_NAMES[segment],
                    "loaded": True,
                    "python_class": (
                        f"{model.__class__.__module__}."
                        f"{model.__class__.__name__}"
                    ),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "segment": segment,
                    "model_file": MODEL_FILE_NAMES[segment],
                    "loaded": False,
                    "python_class": "",
                    "error": str(exc),
                }
            )

    return pd.DataFrame(rows)


def validate_models(models: Mapping[str, Any]) -> None:
    """
    Raise lỗi nếu thiếu một trong ba pipeline notebook P2.
    """
    for segment in SUPPORTED_MODEL_SEGMENTS:
        _, model = get_model_for_segment(models, segment)

        if not hasattr(model, "predict"):
            raise TypeError(
                f"Model `{segment}` không có predict()."
            )
