import joblib
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

FEATURES = [
    "lag_1w", "lag_2w", "lag_3w", "lag_4w", "lag_8w",
    "roll_4w_mean", "roll_8w_mean", "roll_4w_std",
    "month", "week_of_year", "NATURE_CONSOM", "prod_enc",
]

_model = None
_le = None
_ml_df = None
_weekly_df = None
_prod_summary = None
_recommendations = None


def _load():
    global _model, _le, _ml_df, _weekly_df, _prod_summary, _recommendations
    if _model is None:
        _model = joblib.load(BASE_DIR / "models" / "lgbm_model.pkl")
        _le = joblib.load(BASE_DIR / "models" / "label_encoder.pkl")
        _ml_df = pd.read_csv(BASE_DIR / "data" / "ml_weekly_features.csv", parse_dates=["week_start"])
        _weekly_df = pd.read_csv(BASE_DIR / "data" / "weekly_series.csv", parse_dates=["week_start"])
        _prod_summary = pd.read_csv(BASE_DIR / "data" / "product_summary.csv")
        _recommendations = pd.read_csv(BASE_DIR / "data" / "restocking_recommendations.csv")


def get_model():
    _load()
    return _model


def get_encoder():
    _load()
    return _le


def get_ml_df():
    _load()
    return _ml_df


def get_weekly_df():
    _load()
    return _weekly_df


def get_prod_summary():
    _load()
    return _prod_summary


def get_recommendations():
    _load()
    return _recommendations


def predict_product(code: str, horizon: int = 4) -> list[dict]:
    _load()

    if code not in _le.classes_:
        return []

    prod_df = _ml_df[_ml_df["CODE_PROD"] == code].sort_values("week_start")
    if len(prod_df) < 4:
        return []

    nature = float(prod_df["NATURE_CONSOM"].iloc[-1])
    lib = prod_df["LIB_PROD"].iloc[-1]
    prod_enc = int(_le.transform([code])[0])
    last_date = prod_df["week_start"].max()
    history = list(prod_df["QTE_T"].values)

    results = []
    for h in range(1, horizon + 1):
        forecast_date = last_date + pd.Timedelta(weeks=h)
        n = len(history)
        row = {
            "lag_1w": history[-1] if n >= 1 else 0,
            "lag_2w": history[-2] if n >= 2 else 0,
            "lag_3w": history[-3] if n >= 3 else 0,
            "lag_4w": history[-4] if n >= 4 else 0,
            "lag_8w": history[-8] if n >= 8 else 0,
            "roll_4w_mean": np.mean(history[-4:]) if n >= 2 else history[-1],
            "roll_8w_mean": np.mean(history[-8:]) if n >= 4 else np.mean(history),
            "roll_4w_std": np.std(history[-4:]) if n >= 2 else 0.0,
            "month": int(forecast_date.month),
            "week_of_year": int(forecast_date.isocalendar().week),
            "NATURE_CONSOM": nature,
            "prod_enc": prod_enc,
        }
        X = pd.DataFrame([row])[FEATURES]
        pred = max(0.0, float(_model.predict(X)[0]))
        history.append(pred)
        results.append({
            "forecast_week": forecast_date.strftime("%Y-%m-%d"),
            "horizon_weeks": h,
            "predicted_qty": round(pred, 2),
        })

    return results
