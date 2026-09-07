"""
Validasi input mentah sebelum masuk ke pipeline model.
"""

import pandas as pd

RAW_REQUIRED_COLS = [
    "Make", "Type", "Year", "Origin", "Color", "Options",
    "Engine_Size", "Fuel_Type", "Gear_Type", "Mileage", "Region",
]


def validate_input(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Validasi dasar sebelum prediksi.

    Returns
    -------
    (is_valid, message)
    """
    missing = [c for c in RAW_REQUIRED_COLS if c not in df.columns]
    if missing:
        return False, f"Kolom hilang: {missing}"

    if df["Year"].isnull().any():
        return False, "Year tidak boleh kosong"
    if ((df["Year"] < 1950) | (df["Year"] > 2030)).any():
        return False, "Year harus dalam rentang wajar (1950-2030)"

    if df["Mileage"].isnull().any():
        return False, "Mileage tidak boleh kosong"
    if (df["Mileage"] < 0).any():
        return False, "Mileage tidak boleh negatif"

    if df["Engine_Size"].isnull().any():
        return False, "Engine_Size tidak boleh kosong"
    if (df["Engine_Size"] <= 0).any():
        return False, "Engine_Size harus lebih besar dari 0"

    if df["Options"].isin(["Standard", "Semi Full", "Full"]).eq(False).any():
        return False, "Options harus salah satu dari: Standard, Semi Full, Full"

    if df["Fuel_Type"].isin(["Gas", "Diesel", "Hybrid"]).eq(False).any():
        return False, "Fuel_Type harus salah satu dari: Gas, Diesel, Hybrid"

    if df["Gear_Type"].isin(["Automatic", "Manual"]).eq(False).any():
        return False, "Gear_Type harus salah satu dari: Automatic, Manual"

    return True, "OK"
