# Medication Demand Prediction API

A REST API that forecasts weekly medication demand and generates restocking recommendations using a LightGBM model trained on 90k+ pharmacy dispensation records.

---

## Quick Start

### With Docker (recommended)

```bash
docker-compose up --build
```

API available at `http://localhost:8000`  
Swagger docs at `http://localhost:8000/docs`

### Without Docker

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Endpoints

### `GET /api/v1/health`
Service status and model info.

```bash
curl http://localhost:8000/api/v1/health
```
```json
{
  "status": "ok",
  "model": "LightGBM (MAE=17.32, RMSE=82.31)",
  "n_products_eligible": 168,
  "data_range": "2022-06-01 → 2024-01-30"
}
```

---

### `GET /api/v1/products`
List all products. Filter to forecastable ones with `?eligible_only=true`.

```bash
curl "http://localhost:8000/api/v1/products?eligible_only=true"
```
```json
{
  "total": 444,
  "eligible": 168,
  "items": [
    {
      "code_prod": "5BB010143",
      "lib_prod": "SERINGUE 10ML",
      "n_weeks": 88,
      "total_qty": 7698.0,
      "avg_weekly_qty": 87.48,
      "eligible_for_forecast": true
    }
  ]
}
```

---

### `GET /api/v1/products/{code_prod}`
Full product detail: history + forecast. Optionally set forecast horizon with `?horizon=N` (1–12 weeks).

```bash
curl "http://localhost:8000/api/v1/products/5BB010143?horizon=4"
```
```json
{
  "code_prod": "5BB010143",
  "lib_prod": "SERINGUE 10ML",
  "nature_consom": 1.0,
  "n_weeks_history": 88,
  "total_qty": 7698.0,
  "avg_weekly_qty": 87.48,
  "first_week": "2022-06-06",
  "last_week": "2024-01-29",
  "history": [{"week_start": "2022-06-06", "qty": 62.0}, "..."],
  "forecast": [
    {"forecast_week": "2024-02-05", "horizon_weeks": 1, "predicted_qty": 36.21},
    "..."
  ]
}
```

---

### `POST /api/v1/predict`
Forecast demand for a specific product.

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"code_prod": "5BB010143", "horizon": 4}'
```
```json
{
  "code_prod": "5BB010143",
  "lib_prod": "SERINGUE 10ML",
  "nature_consom": 1.0,
  "horizon": 4,
  "total_predicted_qty": 162.95,
  "forecasts": [
    {"forecast_week": "2024-02-05", "horizon_weeks": 1, "predicted_qty": 36.21},
    {"forecast_week": "2024-02-12", "horizon_weeks": 2, "predicted_qty": 32.69},
    {"forecast_week": "2024-02-19", "horizon_weeks": 3, "predicted_qty": 46.79},
    {"forecast_week": "2024-02-26", "horizon_weeks": 4, "predicted_qty": 47.26}
  ]
}
```

---

### `GET /api/v1/recommendations`
Restocking priority list. Filter by `?priority=HIGH|MEDIUM|LOW|ALL` and `?limit=N`.

```bash
curl "http://localhost:8000/api/v1/recommendations?priority=HIGH&limit=10"
```
```json
{
  "total_products": 167,
  "high": 42,
  "medium": 42,
  "low": 83,
  "items": [
    {
      "code_prod": "5BB090420",
      "lib_prod": "GANT D'EXAMEN EN LATEX NON POUDRE TAILLE",
      "nature_consom": 0.0,
      "demand_next_4w": 1681.9,
      "safety_stock": 996.3,
      "recommended_qty": 2678,
      "restock_priority": "HIGH"
    }
  ]
}
```

---

## Model Details

| Property | Value |
|---|---|
| Model | LightGBM (global, all products) |
| Target | Weekly quantity (QTE_T) |
| Loss | MAE (L1) |
| MAE | 17.32 units/week |
| RMSE | 82.31 units/week |
| Baseline MAE | 20.46 (rolling 4w avg) |
| Improvement | -15.4% vs baseline |
| Training data | Jun 2022 – Dec 2023 |
| Test data | Jan 2024 |
| Eligible products | 168 (≥ 20 weeks history) |

**Features:** lag_1w, lag_2w, lag_3w, lag_4w, lag_8w, roll_4w_mean, roll_8w_mean, roll_4w_std, month, week_of_year, NATURE_CONSOM, product_encoding

**Restocking logic:**
- `safety_stock = 1.65 × weekly_std × √(lead_time=2 weeks)` — 95% service level
- `recommended_qty = predicted_demand_4w + safety_stock`
- Priority: HIGH (top 25% demand), MEDIUM (25–50%), LOW (bottom 50%)

---

## Project Structure

```
api/
├── app/
│   ├── main.py        # FastAPI app + lifespan
│   ├── routes.py      # All endpoint handlers
│   ├── schemas.py     # Pydantic request/response models
│   └── model.py       # Model loader + prediction engine
├── models/
│   ├── lgbm_model.pkl
│   └── label_encoder.pkl
├── data/
│   ├── ml_weekly_features.csv
│   ├── weekly_series.csv
│   ├── product_summary.csv
│   └── restocking_recommendations.csv
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
