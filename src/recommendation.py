# -*- coding: utf-8 -*-
"""
src/recommendation.py

Sinh khuyến nghị cho người dùng dựa trên:
- Giá đề xuất từ predictor.py.
- Giá rao thực tế.
- Kết quả phát hiện bất thường từ anomaly_detector.py.
- Thông tin xe và các cờ tín hiệu P2.

API được app.py import:
- generate_recommendation()
- generate_batch_recommendations()

Khuyến nghị không khẳng định chất lượng thực tế của xe.
Nội dung nhấn mạnh việc kiểm tra giấy tờ, số khung/số máy,
tình trạng máy móc, ngập nước, đâm đụng và lịch sử bảo dưỡng.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


# ============================================================================
# HẰNG SỐ
# ============================================================================

NORMAL_TOLERANCE_PCT = 10.0
MODERATE_DEVIATION_PCT = 20.0
HIGH_DEVIATION_PCT = 35.0

HIGH_ANOMALY_SCORE = 85.0
VERY_HIGH_ANOMALY_SCORE = 95.0

SAFE_PRICE_BUFFER_LOW = 0.90
SAFE_PRICE_BUFFER_HIGH = 1.10


# ============================================================================
# HÀM HỖ TRỢ
# ============================================================================

def _to_million(value: Any) -> float:
    if value is None or pd.isna(value):
        return np.nan

    number = float(value)

    if abs(number) > 100_000:
        number /= 1_000_000

    return number


def _format_million(value: Any) -> str:
    number = _to_million(value)

    if pd.isna(number):
        return "không xác định"

    return (
        f"{number:,.1f} triệu đồng"
        .replace(",", ".")
    )


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)

    if not np.isfinite(number):
        return float(default)

    return number


def _safe_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if pd.isna(value):
        return False

    text = str(value).strip().lower()

    return text in {
        "true",
        "1",
        "yes",
        "y",
        "có",
        "co",
        "bất thường",
    }


def _vehicle_description(
    vehicle_info: Mapping[str, Any] | None,
) -> str:
    if not vehicle_info:
        return "xe này"

    brand = str(
        vehicle_info.get("Thương hiệu", "")
    ).strip()
    model = str(
        vehicle_info.get("Dòng xe", "")
    ).strip()
    year = vehicle_info.get("Năm đăng ký")

    parts = [
        value
        for value in [brand, model]
        if value and value.lower() not in {"nan", "none"}
    ]

    description = " ".join(parts) or "xe này"

    if year is not None and not pd.isna(year):
        description += f" đăng ký {year}"

    return description


def _extract_signal_details(
    anomaly_result: Mapping[str, Any],
) -> list[str]:
    details: list[str] = []

    flag_resid = str(
        anomaly_result.get("flag_resid", "")
    )
    flag_minmax = str(
        anomaly_result.get("flag_minmax", "")
    )
    flag_range = str(
        anomaly_result.get("flag_range", "")
    )
    flag_iso = str(
        anomaly_result.get("flag_iso", "")
    )

    resid_z = _safe_float(
        anomaly_result.get("resid_z"),
        default=0.0,
    )

    if flag_resid not in {"", "Bình thường"}:
        details.append(
            f"Residual-z được gắn cờ **{flag_resid}** "
            f"(z = {resid_z:.2f})."
        )

    if flag_minmax not in {"", "Bình thường"}:
        p1 = anomaly_result.get("GiaMin_P1")
        p99 = anomaly_result.get("GiaMax_P99")

        details.append(
            "Giá rao nằm ngoài khoảng P1/P99 của nhóm tham chiếu "
            f"({_format_million(p1)} – {_format_million(p99)})."
        )

    if flag_range not in {"", "Bình thường"}:
        p10 = anomaly_result.get("P10")
        p90 = anomaly_result.get("P90")

        details.append(
            "Giá rao nằm ngoài vùng P10/P90 phổ biến của nhóm "
            f"({_format_million(p10)} – {_format_million(p90)})."
        )

    if flag_iso == "Bất thường":
        details.append(
            "Isolation Forest nhận diện tổ hợp tuổi xe, số km và giá "
            "khác đáng kể so với phần lớn tin tham chiếu."
        )

    if _safe_bool(
        anomaly_result.get("used_fallback_group")
    ):
        details.append(
            "Nhóm Thương hiệu + Dung tích có ít hơn 10 mẫu; "
            "hệ thống đã fallback sang PhanKhucXe theo đúng P2."
        )

    return details


def _inspection_checklist(
    vehicle_info: Mapping[str, Any] | None,
) -> list[str]:
    checklist = [
        "Đối chiếu cavet, số khung, số máy và danh tính người bán.",
        "Kiểm tra máy nguội, tiếng máy, khói xả, rò rỉ dầu và hệ thống điện.",
        "Kiểm tra dấu hiệu ngập nước, đâm đụng, hàn/chỉnh khung sườn.",
        "Chạy thử xe và kiểm tra phanh, lốp, giảm xóc, tay lái.",
        "Chỉ đặt cọc sau khi xác minh xe và giấy tờ trực tiếp.",
    ]

    if not vehicle_info:
        return checklist

    km = vehicle_info.get("Số Km đã đi")
    if km is not None and not pd.isna(km):
        try:
            km_value = float(km)
            if km_value == 999_999:
                checklist.insert(
                    1,
                    "Số km không rõ: cần đánh giá độ mòn tay lái, gác chân, "
                    "phanh, lốp và lịch sử bảo dưỡng.",
                )
        except (TypeError, ValueError):
            pass

    text = (
        f"{vehicle_info.get('Tiêu đề', '')} "
        f"{vehicle_info.get('Mô tả chi tiết', '')}"
    ).lower()

    if "chính chủ" in text:
        checklist.insert(
            1,
            "Tin có ghi 'chính chủ'; vẫn cần đối chiếu tên trên cavet "
            "với giấy tờ tùy thân.",
        )

    if "zin" in text or "nguyên bản" in text:
        checklist.insert(
            2,
            "Tin có ghi xe zin/nguyên bản; nên kiểm tra chi tiết ốc máy, "
            "dàn điện, khung sườn và phụ tùng.",
        )

    return checklist


# ============================================================================
# SINGLE RECOMMENDATION
# ============================================================================

def generate_recommendation(
    predicted_price: float,
    asking_price: float,
    anomaly_result: Mapping[str, Any],
    vehicle_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Tạo recommendation dictionary đúng cách app.py đang hiển thị:

    {
        "level": "success" | "info" | "warning" | "error",
        "message": "...",
        "details": ["...", ...],
    }
    """
    predicted = _to_million(predicted_price)
    asking = _to_million(asking_price)

    if pd.isna(predicted) or predicted <= 0:
        raise ValueError(
            "predicted_price phải là số dương hợp lệ."
        )

    if pd.isna(asking) or asking <= 0:
        raise ValueError(
            "asking_price phải là số dương hợp lệ."
        )

    vehicle = _vehicle_description(vehicle_info)

    residual = asking - predicted
    deviation_pct = residual / predicted * 100

    anomaly_score = _safe_float(
        anomaly_result.get("anomaly_score"),
        default=0.0,
    )
    threshold = _safe_float(
        anomaly_result.get("threshold"),
        default=95.0,
    )
    is_anomaly = _safe_bool(
        anomaly_result.get("is_anomaly")
    )

    final_label = str(
        anomaly_result.get(
            "flag_final",
            anomaly_result.get("label", "Bình thường"),
        )
    )

    lower_reference = predicted * SAFE_PRICE_BUFFER_LOW
    upper_reference = predicted * SAFE_PRICE_BUFFER_HIGH

    details = [
        (
            f"Giá đề xuất: **{_format_million(predicted)}**; "
            f"giá rao: **{_format_million(asking)}**."
        ),
        (
            f"Chênh lệch: **{residual:+.1f} triệu đồng "
            f"({deviation_pct:+.1f}%)**."
        ),
        (
            "Khoảng tham khảo ±10% quanh giá đề xuất: "
            f"**{_format_million(lower_reference)} – "
            f"{_format_million(upper_reference)}**."
        ),
        (
            f"Điểm bất thường: **{anomaly_score:.1f}/100**; "
            f"ngưỡng hiện tại: **{threshold:.1f}/100**."
        ),
    ]

    details.extend(
        _extract_signal_details(anomaly_result)
    )

    if is_anomaly and residual < 0:
        level = "error"
        message = (
            f"{vehicle.capitalize()} có mức giá thấp bất thường so với "
            "giá đề xuất. Không nên xem đây là cơ hội mua ngay; "
            "cần kiểm tra kỹ rủi ro về giấy tờ và tình trạng xe."
        )

        details.extend(
            [
                "Yêu cầu xem xe trực tiếp; tránh chuyển tiền/cọc trước.",
                "Xác minh xe không bị tranh chấp, cầm cố, mất cắp hoặc chỉ bán giấy.",
                "Kiểm tra nguyên nhân giá thấp: cần tiền gấp, xe lỗi, xe ngập, "
                "tai nạn, giấy tờ không hợp lệ hoặc tin không bán xe nguyên chiếc.",
            ]
        )

    elif is_anomaly and residual > 0:
        level = "warning"
        message = (
            f"{vehicle.capitalize()} đang được rao cao bất thường so với "
            "giá đề xuất. Nên thương lượng hoặc so sánh thêm các tin cùng nhóm."
        )

        target_offer = min(
            asking,
            predicted * 1.05,
        )

        details.extend(
            [
                (
                    "Có thể bắt đầu thương lượng quanh "
                    f"**{_format_million(target_offer)}**, "
                    "sau khi kiểm tra tình trạng thực tế."
                ),
                "Chỉ chấp nhận mức giá cao nếu xe có hồ sơ bảo dưỡng tốt, "
                "phụ kiện có giá trị, giấy tờ rõ ràng và tình trạng vượt trội.",
            ]
        )

    elif abs(deviation_pct) <= NORMAL_TOLERANCE_PCT:
        level = "success"
        message = (
            f"Giá rao của {vehicle} khá sát giá đề xuất và chưa có dấu hiệu "
            "bất thường mạnh. Có thể tiếp tục kiểm tra xe và thương lượng nhẹ."
        )

    elif residual < 0:
        level = "info"
        message = (
            f"Giá rao của {vehicle} thấp hơn giá đề xuất nhưng chưa vượt "
            "ngưỡng bất thường tổng hợp. Đây có thể là mức giá hấp dẫn, "
            "song vẫn cần kiểm tra nguyên nhân giảm giá."
        )

    else:
        level = "warning"
        message = (
            f"Giá rao của {vehicle} cao hơn giá đề xuất nhưng chưa thuộc "
            "top 5% bất thường. Nên thương lượng và đối chiếu với các xe tương đồng."
        )

    if final_label not in {"", "Bình thường"}:
        details.append(
            f"Kết luận tổng hợp của hệ thống: **{final_label}**."
        )

    details.append("Checklist kiểm tra trước giao dịch:")
    details.extend(
        _inspection_checklist(vehicle_info)
    )

    return {
        "level": level,
        "message": message,
        "details": details,
        "predicted_price": predicted,
        "asking_price": asking,
        "residual": residual,
        "deviation_pct": deviation_pct,
        "anomaly_score": anomaly_score,
        "threshold": threshold,
        "is_anomaly": is_anomaly,
        "label": final_label,
    }


# ============================================================================
# BATCH RECOMMENDATION
# ============================================================================

def _row_price(
    row: pd.Series,
    candidates: list[str],
) -> float:
    for column in candidates:
        if column not in row.index:
            continue

        value = row[column]

        if value is None or pd.isna(value):
            continue

        try:
            return _to_million(value)
        except (TypeError, ValueError):
            continue

    return np.nan


def _batch_recommendation_for_row(
    row: pd.Series,
) -> dict[str, Any]:
    predicted = _row_price(
        row,
        [
            "predicted_price",
            "gia_du_doan",
        ],
    )

    asking = _row_price(
        row,
        [
            "asking_price",
            "Giá_clean",
            "Gia_clean",
            "Gia_rao_clean",
            "Giá rao",
            "gia_rao",
            "Giá",
        ],
    )

    is_anomaly = _safe_bool(
        row.get("is_anomaly", False)
    )

    residual = (
        _safe_float(row.get("residual"))
        if "residual" in row.index
        else (
            asking - predicted
            if pd.notna(asking) and pd.notna(predicted)
            else np.nan
        )
    )

    deviation_pct = (
        residual / predicted * 100
        if pd.notna(residual)
        and pd.notna(predicted)
        and predicted != 0
        else np.nan
    )

    anomaly_score = _safe_float(
        row.get("anomaly_score"),
        default=0.0,
    )

    final_label = str(
        row.get(
            "flag_final",
            row.get("label", "Bình thường"),
        )
    )

    if is_anomaly and pd.notna(residual) and residual < 0:
        code = "REVIEW_CHEAP_ANOMALY"
        level = "error"
        recommendation = (
            "Giá quá thấp bất thường: ưu tiên kiểm tra giấy tờ, "
            "tình trạng xe và tránh đặt cọc trước."
        )

    elif is_anomaly and pd.notna(residual) and residual > 0:
        code = "NEGOTIATE_EXPENSIVE_ANOMALY"
        level = "warning"
        recommendation = (
            "Giá quá cao bất thường: so sánh xe cùng nhóm và thương lượng mạnh."
        )

    elif pd.notna(deviation_pct) and abs(deviation_pct) <= NORMAL_TOLERANCE_PCT:
        code = "PRICE_CLOSE_TO_MODEL"
        level = "success"
        recommendation = (
            "Giá gần mức đề xuất: tiếp tục kiểm tra xe và thương lượng nhẹ."
        )

    elif pd.notna(residual) and residual < 0:
        code = "LOWER_THAN_MODEL"
        level = "info"
        recommendation = (
            "Giá thấp hơn đề xuất nhưng chưa bị gắn cờ: "
            "kiểm tra nguyên nhân giảm giá."
        )

    elif pd.notna(residual) and residual > 0:
        code = "HIGHER_THAN_MODEL"
        level = "warning"
        recommendation = (
            "Giá cao hơn đề xuất: nên thương lượng hoặc tìm thêm tin so sánh."
        )

    else:
        code = "INSUFFICIENT_DATA"
        level = "info"
        recommendation = (
            "Chưa đủ dữ liệu giá để đưa ra khuyến nghị."
        )

    return {
        "recommendation_code": code,
        "recommendation_level": level,
        "recommendation": recommendation,
        "recommendation_label": final_label,
        "recommendation_anomaly_score": anomaly_score,
        "recommendation_deviation_pct": deviation_pct,
    }


def generate_batch_recommendations(
    result_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Bổ sung các cột khuyến nghị cho kết quả batch.

    Không thay đổi số dòng và thứ tự của DataFrame.
    """
    if not isinstance(result_df, pd.DataFrame):
        raise TypeError(
            "result_df phải là pandas DataFrame."
        )

    if result_df.empty:
        output = result_df.copy()

        for column in [
            "recommendation_code",
            "recommendation_level",
            "recommendation",
            "recommendation_label",
            "recommendation_anomaly_score",
            "recommendation_deviation_pct",
        ]:
            output[column] = pd.Series(dtype=object)

        return output

    output = result_df.copy()

    recommendations = output.apply(
        _batch_recommendation_for_row,
        axis=1,
        result_type="expand",
    )

    for column in recommendations.columns:
        output[column] = recommendations[column].to_numpy()

    # Alias tiếng Việt thuận tiện khi export.
    output["Khuyến nghị"] = output["recommendation"]
    output["Mức khuyến nghị"] = output["recommendation_level"]
    output["Mã khuyến nghị"] = output["recommendation_code"]

    return output.reset_index(drop=True)


# ============================================================================
# BÁO CÁO KHuyến NGHỊ BATCH
# ============================================================================

def get_batch_recommendation_summary(
    result_df: pd.DataFrame,
) -> dict[str, Any]:
    if not isinstance(result_df, pd.DataFrame):
        raise TypeError(
            "result_df phải là pandas DataFrame."
        )

    if result_df.empty:
        return {
            "rows": 0,
            "recommendation_counts": {},
            "level_counts": {},
        }

    frame = (
        result_df
        if "recommendation_code" in result_df.columns
        else generate_batch_recommendations(result_df)
    )

    return {
        "rows": int(len(frame)),
        "recommendation_counts": (
            frame["recommendation_code"]
            .value_counts()
            .to_dict()
        ),
        "level_counts": (
            frame["recommendation_level"]
            .value_counts()
            .to_dict()
        ),
        "mean_anomaly_score": float(
            pd.to_numeric(
                frame.get(
                    "anomaly_score",
                    pd.Series(np.nan, index=frame.index),
                ),
                errors="coerce",
            ).mean()
        ),
    }


__all__ = [
    "generate_recommendation",
    "generate_batch_recommendations",
    "get_batch_recommendation_summary",
]
