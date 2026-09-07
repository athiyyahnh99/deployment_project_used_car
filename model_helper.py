"""
Load model (joblib) dan jalankan prediksi harga mobil.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# WAJIB di-import sebelum joblib.load: pipeline yang di-pickle menyimpan
# referensi ke class CarAgeTransformer & FrequencyEncoder.
from custom_transformers import CarAgeTransformer, FrequencyEncoder  # noqa: F401
import joblib

# --- Workaround khusus ---
# Waktu training, CarAgeTransformer & FrequencyEncoder didefinisikan langsung
# di cell notebook (bukan di-import dari custom_transformers.py), jadi
# joblib.dump menyimpan referensi module-nya sebagai "__main__" (nama module
# notebook saat itu). Kalau di-load dari script lain, "__main__" merujuk ke
# script itu sendiri (mis. streamlit_app.py) yang tidak punya definisi class
# tersebut -> AttributeError saat unpickling.
# Fix: suntikkan class ini ke sys.modules['__main__'] SEBELUM joblib.load()
# supaya pickle bisa menemukannya lewat "__main__.CarAgeTransformer" dkk.
sys.modules["__main__"].CarAgeTransformer = CarAgeTransformer
sys.modules["__main__"].FrequencyEncoder = FrequencyEncoder

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
