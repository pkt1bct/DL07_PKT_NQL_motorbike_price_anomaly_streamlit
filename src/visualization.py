# -*- coding: utf-8 -*-
"""
src/visualization.py

Các hàm trực quan hóa dùng bởi app.py.

Module chuyển các biểu đồ trong notebook P2 và phần
P2_Bat_thuong_Gia sang các hàm trả về matplotlib.figure.Figure,
phù hợp với:
    st.pyplot(fig, use_container_width=True)

API được app.py import:
- plot_anomaly_score_components
- plot_brand_distribution
- plot_correlation_heatmap
- plot_district_distribution
- plot_model_metrics
- plot_prediction_comparison
- plot_price_by_brand
- plot_price_by_vehicle_type
- plot_price_distribution
- plot_segment_distribution
- plot_year_price_scatter
"""

from __future__ import annotations

from typing import Any
from collections.abc import Mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter


# ============================================================================
# CẤU HÌNH CHUNG
# ============================================================================

DEFAULT_FIGSIZE = (10, 6)
PRICE_CANDIDATES = [
    "Gia_clean",
    "Giá_clean",
    "gia_thuc_te",
    "Giá rao",
    "Giá",
]
YEAR_CANDIDATES = [
    "Nam_clean",
    "NamDangKy",
    "Năm đăng ký",
]
DISTRICT_CANDIDATES = [
    "Quan",
    "Quan_clean",
    "Quận",
    "Địa chỉ",
]
SEGMENT_CANDIDATES = [
    "PhanKhucXe",
    "Segment",
    "Phân khúc",
]


# ============================================================================
# HÀM HỖ TRỢ
# ============================================================================

def _ensure_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Đầu vào phải là pandas DataFrame.")

    if df.empty:
        raise ValueError("DataFrame không có dữ liệu để vẽ.")

    return df.copy()


def _first_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _price_series(df: pd.DataFrame) -> pd.Series:
    column = _first_existing_column(df, PRICE_CANDIDATES)

    if column is None:
        raise ValueError(
            "Không tìm thấy cột giá. "
            f"Cần một trong: {PRICE_CANDIDATES}"
        )

    values = pd.to_numeric(df[column], errors="coerce").astype(float)

    # Nếu giá còn ở đơn vị VND thì chuyển sang triệu đồng.
    mask = values.abs() > 100_000
    values.loc[mask] = values.loc[mask] / 1_000_000

    return values


def _year_series(df: pd.DataFrame) -> pd.Series:
    column = _first_existing_column(df, YEAR_CANDIDATES)

    if column is None:
        raise ValueError(
            "Không tìm thấy cột năm đăng ký."
        )

    values = pd.to_numeric(df[column], errors="coerce")
    return values


def _district_series(df: pd.DataFrame) -> pd.Series:
    column = _first_existing_column(df, DISTRICT_CANDIDATES)

    if column is None:
        raise ValueError(
            "Không tìm thấy cột quận/huyện hoặc địa chỉ."
        )

    values = df[column].fillna("Không rõ").astype(str).str.strip()

    if column == "Địa chỉ":
        values = values.str.split(",").str[0].str.strip()

    return values.replace("", "Không rõ")


def _segment_series(df: pd.DataFrame) -> pd.Series:
    column = _first_existing_column(df, SEGMENT_CANDIDATES)

    if column is None:
        raise ValueError(
            "Không tìm thấy cột phân khúc."
        )

    return (
        df[column]
        .fillna("Không rõ")
        .astype(str)
        .str.strip()
        .replace("", "Không rõ")
    )


def _new_figure(
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
) -> tuple[Figure, Any]:
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _finish(
    fig: Figure,
    ax: Any,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    grid_axis: str | None = "y",
) -> Figure:
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if grid_axis:
        ax.grid(
            axis=grid_axis,
            alpha=0.22,
            linewidth=0.7,
        )

    fig.tight_layout()
    return fig


def _annotate_bars(
    ax: Any,
    decimals: int = 0,
) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        width = patch.get_width()

        if not np.isfinite(height):
            continue

        if width >= height and patch.get_y() != 0:
            value = width
            x = width
            y = patch.get_y() + patch.get_height() / 2
            ax.annotate(
                f"{value:.{decimals}f}",
                (x, y),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
            )
        else:
            value = height
            x = patch.get_x() + patch.get_width() / 2
            y = height
            ax.annotate(
                f"{value:.{decimals}f}",
                (x, y),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )


# ============================================================================
# EDA
# ============================================================================

def plot_brand_distribution(
    df: pd.DataFrame,
    top_n: int = 15,
) -> Figure:
    data = _ensure_dataframe(df)

    if "Thương hiệu" not in data.columns:
        raise ValueError("Thiếu cột `Thương hiệu`.")

    counts = (
        data["Thương hiệu"]
        .fillna("Không rõ")
        .astype(str)
        .value_counts()
        .head(top_n)
        .sort_values()
    )

    fig, ax = _new_figure((10, 6))
    ax.barh(counts.index, counts.values)
    _annotate_bars(ax)

    return _finish(
        fig,
        ax,
        "Phân bố tin đăng theo thương hiệu",
        xlabel="Số tin đăng",
        ylabel="Thương hiệu",
        grid_axis="x",
    )


def plot_district_distribution(
    df: pd.DataFrame,
    top_n: int = 15,
) -> Figure:
    data = _ensure_dataframe(df)
    districts = _district_series(data)

    counts = districts.value_counts().head(top_n).sort_values()

    fig, ax = _new_figure((10, 6))
    ax.barh(counts.index, counts.values)
    _annotate_bars(ax)

    return _finish(
        fig,
        ax,
        f"Top {min(top_n, len(counts))} Quận/Huyện có nhiều tin đăng",
        xlabel="Số tin đăng",
        ylabel="Quận/Huyện",
        grid_axis="x",
    )


def plot_price_distribution(
    df: pd.DataFrame,
    bins: int = 50,
) -> Figure:
    data = _ensure_dataframe(df)
    prices = _price_series(data).dropna()

    if prices.empty:
        raise ValueError("Không có giá hợp lệ để vẽ.")

    upper = prices.quantile(0.99)
    display_values = prices[prices <= upper]

    fig, ax = _new_figure((10, 6))
    ax.hist(
        display_values,
        bins=bins,
        edgecolor="white",
        linewidth=0.4,
        alpha=0.85,
    )

    median = display_values.median()
    ax.axvline(
        median,
        linestyle="--",
        linewidth=1.8,
        label=f"Median = {median:.1f} triệu",
    )
    ax.legend(frameon=False)

    return _finish(
        fig,
        ax,
        "Phân phối giá xe máy cũ (đến percentile 99%)",
        xlabel="Giá (triệu đồng)",
        ylabel="Số tin đăng",
    )


def plot_price_by_brand(
    df: pd.DataFrame,
    top_n: int = 12,
) -> Figure:
    data = _ensure_dataframe(df)

    if "Thương hiệu" not in data.columns:
        raise ValueError("Thiếu cột `Thương hiệu`.")

    temp = pd.DataFrame(
        {
            "brand": data["Thương hiệu"].fillna("Không rõ").astype(str),
            "price": _price_series(data),
        }
    ).dropna()

    top_brands = (
        temp["brand"]
        .value_counts()
        .head(top_n)
        .index
    )

    temp = temp[temp["brand"].isin(top_brands)]
    order = (
        temp.groupby("brand")["price"]
        .median()
        .sort_values()
        .index
        .tolist()
    )

    groups = [
        temp.loc[temp["brand"] == brand, "price"].to_numpy()
        for brand in order
    ]

    fig, ax = _new_figure((12, 6))
    ax.boxplot(
        groups,
        labels=order,
        showfliers=False,
        patch_artist=True,
    )
    ax.tick_params(axis="x", rotation=35)

    return _finish(
        fig,
        ax,
        "Phân bố giá theo thương hiệu",
        xlabel="Thương hiệu",
        ylabel="Giá (triệu đồng)",
    )


def plot_price_by_vehicle_type(
    df: pd.DataFrame,
) -> Figure:
    data = _ensure_dataframe(df)

    if "Loại xe" not in data.columns:
        raise ValueError("Thiếu cột `Loại xe`.")

    temp = pd.DataFrame(
        {
            "vehicle_type": (
                data["Loại xe"]
                .fillna("Không rõ")
                .astype(str)
            ),
            "price": _price_series(data),
        }
    ).dropna()

    order = (
        temp.groupby("vehicle_type")["price"]
        .median()
        .sort_values()
        .index
        .tolist()
    )

    groups = [
        temp.loc[
            temp["vehicle_type"] == vehicle_type,
            "price",
        ].to_numpy()
        for vehicle_type in order
    ]

    fig, ax = _new_figure((10, 6))
    ax.boxplot(
        groups,
        labels=order,
        showfliers=False,
        patch_artist=True,
    )
    ax.tick_params(axis="x", rotation=25)

    return _finish(
        fig,
        ax,
        "Phân bố giá theo loại xe",
        xlabel="Loại xe",
        ylabel="Giá (triệu đồng)",
    )


def plot_year_price_scatter(
    df: pd.DataFrame,
) -> Figure:
    data = _ensure_dataframe(df)

    temp = pd.DataFrame(
        {
            "year": _year_series(data),
            "price": _price_series(data),
        }
    ).dropna()

    if temp.empty:
        raise ValueError("Không có dữ liệu năm/giá hợp lệ.")

    fig, ax = _new_figure((10, 6))
    ax.scatter(
        temp["year"],
        temp["price"],
        alpha=0.35,
        s=16,
    )

    if len(temp) >= 2:
        coefficients = np.polyfit(
            temp["year"],
            temp["price"],
            deg=1,
        )
        x_line = np.linspace(
            temp["year"].min(),
            temp["year"].max(),
            100,
        )
        y_line = (
            coefficients[0] * x_line
            + coefficients[1]
        )
        ax.plot(
            x_line,
            y_line,
            linestyle="--",
            linewidth=1.5,
            label="Xu hướng tuyến tính",
        )
        ax.legend(frameon=False)

    return _finish(
        fig,
        ax,
        "Quan hệ giữa năm đăng ký và giá",
        xlabel="Năm đăng ký",
        ylabel="Giá (triệu đồng)",
    )


def plot_segment_distribution(
    df: pd.DataFrame,
) -> Figure:
    data = _ensure_dataframe(df)
    segments = _segment_series(data)

    counts = segments.value_counts()

    fig, ax = _new_figure((9, 6))
    ax.bar(counts.index, counts.values)
    ax.tick_params(axis="x", rotation=25)
    _annotate_bars(ax)

    return _finish(
        fig,
        ax,
        "Phân bố tin đăng theo phân khúc xe",
        xlabel="Phân khúc",
        ylabel="Số tin đăng",
    )


def plot_correlation_heatmap(
    df: pd.DataFrame,
    max_features: int = 16,
) -> Figure:
    data = _ensure_dataframe(df)

    numeric = data.select_dtypes(
        include=[np.number]
    ).copy()

    if numeric.shape[1] < 2:
        raise ValueError(
            "Cần ít nhất hai biến số để tính tương quan."
        )

    # Ưu tiên các cột có phương sai và dữ liệu đầy đủ hơn.
    valid_count = numeric.notna().sum()
    variance = numeric.var(numeric_only=True)

    score = (
        valid_count.rank(pct=True)
        + variance.rank(pct=True)
    )

    selected = (
        score.sort_values(ascending=False)
        .head(max_features)
        .index
    )

    corr = numeric[selected].corr()

    fig, ax = _new_figure((11, 9))
    image = ax.imshow(
        corr.to_numpy(),
        vmin=-1,
        vmax=1,
        cmap="RdYlBu_r",
        aspect="auto",
    )

    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(
        corr.columns,
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_yticklabels(
        corr.index,
        fontsize=8,
    )

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            value = corr.iloc[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Hệ số tương quan Pearson")

    return _finish(
        fig,
        ax,
        "Ma trận tương quan giữa các biến số",
        grid_axis=None,
    )


# ============================================================================
# MODEL / PREDICTION
# ============================================================================

def plot_model_metrics(
    metrics_df: pd.DataFrame,
) -> Figure:
    data = _ensure_dataframe(metrics_df)

    segment_column = _first_existing_column(
        data,
        [
            "Phân khúc",
            "PhanKhuc",
            "segment",
            "Segment",
            "model",
            "Model",
        ],
    )

    if segment_column is None:
        labels = data.index.astype(str)
    else:
        labels = data[segment_column].astype(str)

    metric_candidates = [
        "R2",
        "R²",
        "r2",
        "MAE",
        "RMSE",
        "MAPE",
    ]

    metric_columns = [
        column
        for column in metric_candidates
        if column in data.columns
    ]

    if not metric_columns:
        numeric_columns = (
            data.select_dtypes(include=[np.number])
            .columns
            .tolist()
        )
        metric_columns = numeric_columns[:4]

    if not metric_columns:
        raise ValueError(
            "Không tìm thấy cột metrics dạng số."
        )

    plot_data = data[metric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    x = np.arange(len(data))
    width = 0.8 / len(metric_columns)

    fig, ax = _new_figure((11, 6))

    for idx, column in enumerate(metric_columns):
        offset = (
            idx - (len(metric_columns) - 1) / 2
        ) * width

        ax.bar(
            x + offset,
            plot_data[column],
            width=width,
            label=column,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.legend(frameon=False)

    return _finish(
        fig,
        ax,
        "So sánh metrics giữa các mô hình/phân khúc",
        xlabel="Mô hình / Phân khúc",
        ylabel="Giá trị metric",
    )


def plot_prediction_comparison(
    predicted_price: float,
    asking_price: float,
) -> Figure:
    predicted = float(predicted_price)
    asking = float(asking_price)

    if abs(predicted) > 100_000:
        predicted /= 1_000_000

    if abs(asking) > 100_000:
        asking /= 1_000_000

    values = [predicted, asking]
    labels = ["Giá đề xuất", "Giá rao"]

    fig, ax = _new_figure((8, 5))
    colors = [
        "#4E79A7",
        "#F28E2B",
        "#59A14F",
        "#E15759",
    ]

    bars = ax.bar(
        labels,
        values,
        color=colors,
    )

    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.1f} triệu",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
            ),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    difference = asking - predicted
    difference_pct = (
        difference / predicted * 100
        if predicted
        else 0.0
    )

    ax.text(
        0.5,
        0.96,
        (
            f"Chênh lệch: {difference:+.1f} triệu "
            f"({difference_pct:+.1f}%)"
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )

    return _finish(
        fig,
        ax,
        "So sánh giá rao và giá đề xuất",
        ylabel="Giá (triệu đồng)",
    )


# ============================================================================
# ANOMALY VISUALIZATION
# ============================================================================

def plot_anomaly_score_components(
    components: Mapping[str, Any] | dict[str, Any],
    threshold: float | None = None,
) -> Figure:
    if not isinstance(components, Mapping):
        raise TypeError(
            "components phải là dictionary/Mapping."
        )

    if not components:
        raise ValueError(
            "components không có dữ liệu."
        )

    labels = [str(key) for key in components.keys()]
    values = [
        float(value)
        for value in components.values()
    ]

    fig, ax = _new_figure((9, 5.5))
    bars = ax.bar(labels, values)

    if threshold is not None:
        ax.axhline(
            y=threshold,
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            color="tab:red",
            label=f"Ngưỡng cảnh báo ({threshold:.1f})",
    )

    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.1f}",
            (
                bar.get_x() + bar.get_width() / 2,
                value,
            ),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )

    ax.set_ylim(0, max(100, max(values) * 1.15))
    ax.tick_params(axis="x", rotation=20)
    if threshold is not None:
        ax.legend(frameon=False)

    return _finish(
        fig,
        ax,
        "Điểm của từng tín hiệu trong mô hình phát hiện bất thường",
        xlabel="Tín hiệu",
        ylabel="Percentile score (0–100)",
    )


def plot_anomaly_signal_correlation(
    scored_df: pd.DataFrame,
) -> Figure:
    """
    Biểu đồ đúng phần P2_Bat_thuong_Gia:
    tương quan score_resid, score_minmax, score_range, score_iso.
    """
    data = _ensure_dataframe(scored_df)

    columns = [
        "score_resid",
        "score_minmax",
        "score_range",
        "score_iso",
    ]

    missing = [
        column
        for column in columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "Thiếu cột: " + ", ".join(missing)
        )

    corr = data[columns].corr()

    labels = [
        "Residual-z\n(có giám sát)",
        "Vi phạm\nP1/P99",
        "Ngoài\nP10/P90",
        "Isolation Forest\n(không giám sát)",
    ]

    fig, ax = _new_figure((8, 7))
    image = ax.imshow(
        corr.to_numpy(),
        vmin=-1,
        vmax=1,
        cmap="RdYlBu_r",
    )

    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticklabels(labels)

    for i in range(4):
        for j in range(4):
            ax.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=11,
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Hệ số tương quan Pearson")

    return _finish(
        fig,
        ax,
        "Ma trận tương quan giữa 4 tín hiệu bất thường",
        grid_axis=None,
    )


def plot_anomaly_score_distribution(
    scored_df: pd.DataFrame,
) -> Figure:
    data = _ensure_dataframe(scored_df)

    required = [
        "anomaly_score",
        "is_anomaly",
        "threshold",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "Thiếu cột: " + ", ".join(missing)
        )

    score = pd.to_numeric(
        data["anomaly_score"],
        errors="coerce",
    ).dropna()

    anomaly_values = pd.to_numeric(
        data.loc[
            data["is_anomaly"].astype(bool),
            "anomaly_score",
        ],
        errors="coerce",
    ).dropna()

    threshold = float(
        pd.to_numeric(
            data["threshold"],
            errors="coerce",
        ).dropna().iloc[0]
    )

    fig, ax = _new_figure((10, 6))
    ax.hist(score, bins=60, edgecolor="white", linewidth=0.3)
    ax.hist(
        anomaly_values,
        bins=15,
        edgecolor="white",
        linewidth=0.3,
        alpha=0.85,
        label=f"Bất thường (n={len(anomaly_values)})",
    )
    ax.axvline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=f"Ngưỡng top 5% = {threshold:.1f}",
    )
    ax.legend(frameon=False)

    return _finish(
        fig,
        ax,
        "Phân phối điểm bất thường và ngưỡng phân loại",
        xlabel="Anomaly score (0–100)",
        ylabel="Số tin đăng",
    )


def plot_actual_vs_predicted_anomalies(
    scored_df: pd.DataFrame,
) -> Figure:
    data = _ensure_dataframe(scored_df)

    actual_column = _first_existing_column(
        data,
        ["gia_thuc_te", "Giá_clean", "Gia_clean"],
    )
    predicted_column = _first_existing_column(
        data,
        ["gia_du_doan", "predicted_price"],
    )

    if (
        actual_column is None
        or predicted_column is None
        or "is_anomaly" not in data.columns
    ):
        raise ValueError(
            "Thiếu giá thực, giá dự đoán hoặc is_anomaly."
        )

    actual = pd.to_numeric(
        data[actual_column],
        errors="coerce",
    )
    predicted = pd.to_numeric(
        data[predicted_column],
        errors="coerce",
    )
    anomaly = data["is_anomaly"].fillna(False).astype(bool)

    valid = actual.notna() & predicted.notna()
    actual = actual[valid]
    predicted = predicted[valid]
    anomaly = anomaly[valid]

    fig, ax = _new_figure((9, 6))

    ax.scatter(
        predicted[~anomaly],
        actual[~anomaly],
        alpha=0.25,
        s=12,
        label="Bình thường",
    )
    ax.scatter(
        predicted[anomaly],
        actual[anomaly],
        alpha=0.8,
        s=22,
        label="Bất thường (top 5%)",
    )

    upper = predicted.quantile(0.995)
    if not np.isfinite(upper) or upper <= 0:
        upper = max(predicted.max(), actual.max())

    ax.plot(
        [0, upper],
        [0, upper],
        linestyle="--",
        linewidth=1.3,
        label="Giá thực = Giá đề xuất",
    )
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.legend(frameon=False)

    return _finish(
        fig,
        ax,
        "Giá thực và giá đề xuất — điểm bất thường",
        xlabel="Giá đề xuất (triệu đồng)",
        ylabel="Giá thực đăng bán (triệu đồng)",
    )


__all__ = [
    "plot_anomaly_score_components",
    "plot_brand_distribution",
    "plot_correlation_heatmap",
    "plot_district_distribution",
    "plot_model_metrics",
    "plot_prediction_comparison",
    "plot_price_by_brand",
    "plot_price_by_vehicle_type",
    "plot_price_distribution",
    "plot_segment_distribution",
    "plot_year_price_scatter",
    "plot_anomaly_signal_correlation",
    "plot_anomaly_score_distribution",
    "plot_actual_vs_predicted_anomalies",
]
