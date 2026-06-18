from fastapi import APIRouter, HTTPException, Query
from typing import Literal

from .schemas import (
    PredictRequest, PredictResponse, WeekForecast,
    ProductDetailResponse, HistoryPoint,
    RecommendationsResponse, RecommendationItem,
    ProductsListResponse, ProductListItem,
    HealthResponse,
)
from .model import (
    get_prod_summary, get_weekly_df, get_ml_df,
    get_recommendations, predict_product,
)

router = APIRouter()


# ── GET /health ───────────────────────────────────────────────
@router.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    summary = get_prod_summary()
    eligible = summary[summary["n_weeks"] >= 20]
    return HealthResponse(
        status="ok",
        model="LightGBM (MAE=17.32, RMSE=82.31)",
        n_products_eligible=len(eligible),
        data_range="2022-06-01 → 2024-01-30",
    )


# ── GET /products ─────────────────────────────────────────────
@router.get("/products", response_model=ProductsListResponse, tags=["Products"])
def list_products(
    eligible_only: bool = Query(False, description="Return only products with ≥20 weeks history")
):
    summary = get_prod_summary()
    if eligible_only:
        summary = summary[summary["n_weeks"] >= 20]

    items = [
        ProductListItem(
            code_prod=str(row.CODE_PROD),
            lib_prod=str(row.LIB_PROD),
            n_weeks=int(row.n_weeks),
            total_qty=round(float(row.total_qty), 1),
            avg_weekly_qty=round(float(row.avg_weekly_qty), 2),
            eligible_for_forecast=bool(row.n_weeks >= 20),
        )
        for row in summary.itertuples()
    ]
    full = get_prod_summary()
    return ProductsListResponse(
        total=len(full),
        eligible=int((full["n_weeks"] >= 20).sum()),
        items=items,
    )


# ── GET /products/{code} ──────────────────────────────────────
@router.get("/products/{code_prod}", response_model=ProductDetailResponse, tags=["Products"])
def product_detail(code_prod: str, horizon: int = Query(4, ge=1, le=12)):
    summary = get_prod_summary()
    row = summary[summary["CODE_PROD"] == code_prod]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Product '{code_prod}' not found.")

    r = row.iloc[0]
    weekly = get_weekly_df()
    hist = weekly[weekly["CODE_PROD"] == code_prod].sort_values("week_start")

    history_points = [
        HistoryPoint(week_start=str(h.week_start)[:10], qty=float(h.QTE_T))
        for h in hist.itertuples()
    ]

    forecasts_raw = predict_product(code_prod, horizon)
    if not forecasts_raw:
        raise HTTPException(
            status_code=422,
            detail=f"Product '{code_prod}' has insufficient history for forecasting (need ≥ 20 weeks).",
        )

    forecasts = [WeekForecast(**f) for f in forecasts_raw]

    ml = get_ml_df()
    prod_ml = ml[ml["CODE_PROD"] == code_prod]
    nature = float(prod_ml["NATURE_CONSOM"].iloc[-1]) if not prod_ml.empty else 0.0

    return ProductDetailResponse(
        code_prod=code_prod,
        lib_prod=str(r.LIB_PROD),
        nature_consom=nature,
        n_weeks_history=int(r.n_weeks),
        total_qty=round(float(r.total_qty), 1),
        avg_weekly_qty=round(float(r.avg_weekly_qty), 2),
        first_week=str(r.first_week)[:10],
        last_week=str(r.last_week)[:10],
        history=history_points,
        forecast=forecasts,
    )


# ── POST /predict ─────────────────────────────────────────────
@router.post("/predict", response_model=PredictResponse, tags=["Forecasting"])
def predict(body: PredictRequest):
    forecasts_raw = predict_product(body.code_prod, body.horizon)
    if not forecasts_raw:
        raise HTTPException(
            status_code=404,
            detail=f"Product '{body.code_prod}' not found or has insufficient history (need ≥ 20 weeks).",
        )

    ml = get_ml_df()
    prod_ml = ml[ml["CODE_PROD"] == body.code_prod]
    lib = str(prod_ml["LIB_PROD"].iloc[-1]) if not prod_ml.empty else "Unknown"
    nature = float(prod_ml["NATURE_CONSOM"].iloc[-1]) if not prod_ml.empty else 0.0

    forecasts = [WeekForecast(**f) for f in forecasts_raw]
    total = round(sum(f.predicted_qty for f in forecasts), 2)

    return PredictResponse(
        code_prod=body.code_prod,
        lib_prod=lib,
        nature_consom=nature,
        horizon=body.horizon,
        forecasts=forecasts,
        total_predicted_qty=total,
    )


# ── GET /recommendations ──────────────────────────────────────
@router.get("/recommendations", response_model=RecommendationsResponse, tags=["Restocking"])
def recommendations(
    priority: Literal["HIGH", "MEDIUM", "LOW", "ALL"] = Query("ALL", description="Filter by priority"),
    limit: int = Query(50, ge=1, le=500, description="Max items returned"),
):
    rec = get_recommendations()
    if priority != "ALL":
        rec = rec[rec["restock_priority"] == priority]

    rec = rec.head(limit)

    items = [
        RecommendationItem(
            code_prod=str(r.CODE_PROD),
            lib_prod=str(r.LIB_PROD),
            nature_consom=float(r.NATURE_CONSOM),
            demand_next_4w=round(float(r.demand_next_4w), 1),
            safety_stock=round(float(r.safety_stock), 1),
            recommended_qty=int(r.recommended_qty),
            restock_priority=r.restock_priority,
        )
        for r in rec.itertuples()
    ]

    full = get_recommendations()
    counts = full["restock_priority"].value_counts()
    return RecommendationsResponse(
        total_products=len(full),
        high=int(counts.get("HIGH", 0)),
        medium=int(counts.get("MEDIUM", 0)),
        low=int(counts.get("LOW", 0)),
        items=items,
    )
