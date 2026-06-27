"""Train the SuperKart sales-forecasting pipeline and save one self-contained artifact.

The whole model is a single sklearn Pipeline (preprocessing + XGBoost), so serving just
calls `pipeline.predict` on raw features. No separate scaler, no manual one-hot reindexing
in the API: that brittleness was the main thing wrong with the original deployment.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from .data import CATEGORICAL, FEATURES, NUMERIC, load_xy

SEED = 42
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "superkart_pipeline.pkl"


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    model = XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=SEED,
    )
    return Pipeline([("prep", pre), ("model", model)])


def train(save: bool = True):
    X, y = load_xy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    pipe = build_pipeline().fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    metrics = {
        "r2": float(r2_score(yte, pred)),
        "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
        "mae": float(mean_absolute_error(yte, pred)),
        "mape_pct": float(np.mean(np.abs((yte - pred) / yte)) * 100),
    }
    if save:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump({"pipeline": pipe, "features": FEATURES}, MODEL_PATH)
    return pipe, metrics


if __name__ == "__main__":
    _, m = train()
    print({k: round(v, 3) for k, v in m.items()})
    print("saved:", MODEL_PATH)
