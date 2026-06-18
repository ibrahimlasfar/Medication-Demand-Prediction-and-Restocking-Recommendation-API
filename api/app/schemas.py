from pydantic import BaseModel, Field
from typing import Literal


# ── /predict ──────────────────────────────────────────────────

class PredictRequest(BaseModel):
    code_prod: str = Field(..., description="Product code (e.g. 5BB010143)")
    horizon: int = Field(4, ge=1, le=12, description="Forecast horizon in weeks (1–12)")

    model_config = {"json_schema_extra": {"example": {"code_prod": "5BB010143", "horizon": 4}}}


class WeekForecast(BaseModel):
    forecast_week: str
    horizon_weeks: int
    predicted_qty: float


class PredictResponse(BaseModel):
    code_prod: str
    lib_prod: str
    nature_consom: float
    horizon: int
    forecasts: list[WeekForecast]
    total_predicted_qty: float


# ── /products/{code} ──────────────────────────────────────────

class HistoryPoint(BaseModel):
    week_start: str
    qty: float


class ProductDetailResponse(BaseModel):
    code_prod: str
    lib_prod: str
    nature_consom: float
    n_weeks_history: int
    total_qty: float
    avg_weekly_qty: float
    first_week: str
    last_week: str
    history: list[HistoryPoint]
    forecast: list[WeekForecast]


# ── /recommendations ─────────────────────────────────────────

class RecommendationItem(BaseModel):
    code_prod: str
    lib_prod: str
    nature_consom: float
    demand_next_4w: float
    safety_stock: float
    recommended_qty: int
    restock_priority: Literal["HIGH", "MEDIUM", "LOW"]


class RecommendationsResponse(BaseModel):
    total_products: int
    high: int
    medium: int
    low: int
    items: list[RecommendationItem]


# ── /products ────────────────────────────────────────────────

class ProductListItem(BaseModel):
    code_prod: str
    lib_prod: str
    n_weeks: int
    total_qty: float
    avg_weekly_qty: float
    eligible_for_forecast: bool


class ProductsListResponse(BaseModel):
    total: int
    eligible: int
    items: list[ProductListItem]


# ── /health ──────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model: str
    n_products_eligible: int
    data_range: str
