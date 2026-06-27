"""Shared data prep for training and serving.

Both the training script and the API call `prepare()` so the feature engineering is
identical in both places. This is the single most important thing to get right in a
deployment: training/serving skew is the classic production bug.
"""

from pathlib import Path

import pandas as pd

_CANDIDATES = (Path("../data/SuperKart.csv"), Path("data/SuperKart.csv"))
DATA_PATH = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[-1])

TARGET = "Product_Store_Sales_Total"
REFERENCE_YEAR = 2025

NUMERIC = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Age"]
CATEGORICAL = [
    "Product_Sugar_Content", "Product_Type", "Store_Id",
    "Store_Size", "Store_Location_City_Type", "Store_Type",
]
FEATURES = NUMERIC + CATEGORICAL


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Clean quirks and derive features. Safe to call on training data or one API row."""
    df = df.copy()
    # Known data-entry quirk: 'reg' is the same category as 'Regular'.
    if "Product_Sugar_Content" in df:
        df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace({"reg": "Regular"})
    # Derive store age from establishment year if the raw year was supplied.
    if "Store_Establishment_Year" in df and "Store_Age" not in df:
        df["Store_Age"] = REFERENCE_YEAR - df["Store_Establishment_Year"]
    return df


def load_xy(path: Path = DATA_PATH):
    """Load the full dataset as (X, y)."""
    df = prepare(pd.read_csv(path))
    return df[FEATURES], df[TARGET]


def features_from_request(payload: dict) -> pd.DataFrame:
    """Turn one raw API JSON request into a model-ready single-row frame."""
    df = prepare(pd.DataFrame([payload]))
    return df.reindex(columns=FEATURES)
