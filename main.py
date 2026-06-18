from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .routes import router
from .model import _load


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load model and data on startup
    _load()
    yield


app = FastAPI(
    title="Medication Demand Prediction API",
    description=(
        "Predicts weekly medication demand and generates restocking recommendations "
        "based on historical dispensation data. Built with LightGBM on 90k+ transactions "
        "covering 167 eligible products.\n\n"
        "**Endpoints:**\n"
        "- `GET /health` — service status\n"
        "- `GET /products` — list all products\n"
        "- `GET /products/{code}` — product detail + history + forecast\n"
        "- `POST /predict` — forecast demand for a product\n"
        "- `GET /recommendations` — restocking priority list\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["System"])
def root():
    return {
        "service": "Medication Demand Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
    }
