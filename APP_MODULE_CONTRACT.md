# Hợp đồng hàm mà app.py sử dụng

Để `app.py` chạy được, các module trong `src/` cần cung cấp đúng các hàm sau.

## src/preprocess.py

```python
load_dataset(path: str) -> pd.DataFrame
prepare_eda_data(df: pd.DataFrame) -> pd.DataFrame
get_data_quality_summary(df: pd.DataFrame) -> dict
extract_district(address) -> str
validate_batch_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    optional_columns: list[str],
) -> bool | tuple[bool, str] | dict
```

## src/feature_engineering.py

```python
get_form_options(df: pd.DataFrame) -> dict
build_input_dataframe(values: dict, reference_df: pd.DataFrame) -> pd.DataFrame
```

## src/predictor.py

```python
load_models(
    phothong_path: str,
    trungcap_path: str,
    caocap_path: str,
) -> dict

predict_price(
    input_df: pd.DataFrame,
    models: dict,
    reference_df: pd.DataFrame,
) -> dict

predict_batch(
    batch_df: pd.DataFrame,
    models: dict,
    reference_df: pd.DataFrame,
) -> pd.DataFrame
```

`predict_price()` nên trả về tối thiểu:

```python
{
    "predicted_price": float,
    "lower_bound": float,
    "upper_bound": float,
    "segment": str,
    "model_name": str,
}
```

## src/anomaly_detector.py

```python
analyze_price_anomaly(
    input_df: pd.DataFrame,
    asking_price: float,
    prediction_result: dict,
    reference_df: pd.DataFrame,
) -> dict

analyze_batch_anomalies(
    prediction_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> pd.DataFrame
```

`analyze_price_anomaly()` nên trả về tối thiểu:

```python
{
    "predicted_price": float,
    "anomaly_score": float,
    "threshold": float,
    "is_anomaly": bool,
    "label": str,
    "components": {
        "Residual-z": float,
        "P1/P99": float,
        "P10/P90": float,
        "Isolation Forest": float,
    },
}
```

## src/visualization.py

Mỗi hàm trả về một `matplotlib.figure.Figure`.

```python
plot_brand_distribution(df)
plot_district_distribution(df)
plot_price_distribution(df)
plot_price_by_brand(df)
plot_price_by_vehicle_type(df)
plot_year_price_scatter(df)
plot_correlation_heatmap(df)
plot_segment_distribution(df)
plot_model_metrics(metrics_df)
plot_prediction_comparison(predicted_price, asking_price)
plot_anomaly_score_components(components)
```

## src/recommendation.py

```python
generate_recommendation(
    predicted_price: float,
    asking_price: float,
    anomaly_result: dict,
    vehicle_info: dict,
) -> str | dict

generate_batch_recommendations(df: pd.DataFrame) -> pd.DataFrame
```
