"""
Load model (joblib) dan jalankan prediksi harga mobil.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# WAJIB di-import sebelum joblib.load: pipeline yang di-pickle menyimpan
# referensi ke class CarAgeTransformer & FrequencyEncoder. Tanpa ini,
# joblib.load akan raise AttributeError.
from custom_transformers import CarAgeTransformer, FrequencyEncoder  # noqa: F401
import joblib

MODEL_PATH = Path(__file__).resolve().parent / "models" / "final_model_catboost.joblib"

# Urutan/nama kolom persis seperti saat model di-training (X_train sebelum pipeline)
FEATURE_ORDER = [
    "Make", "Type", "Year", "Origin", "Color", "Options",
    "Engine_Size", "Fuel_Type", "Gear_Type", "Mileage", "Region",
]


def load_model():
    """Load pipeline model (feature_engineering + preprocessing + CatBoost)."""
    return joblib.load(MODEL_PATH)


def predict_price(model, raw_df: pd.DataFrame) -> np.ndarray:
    """
    Prediksi harga mobil (dalam SAR) dari data mentah.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Model hasil load_model().
    raw_df : pd.DataFrame
        Data mentah dengan kolom sesuai RAW_REQUIRED_COLS di validation.py.

    Returns
    -------
    price_sar : np.ndarray
        Prediksi harga dalam SAR (sudah di-inverse dari log1p).
    """
    df = raw_df[FEATURE_ORDER].copy()

    pred_log = model.predict(df)
    price_sar = np.expm1(pred_log)

    return price_sar
