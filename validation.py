"""
validation.py

Validasi input sebelum prediksi (misal: cek Year <= 2022 karena
CarAgeTransformer pakai reference_year=2021, cek Mileage >= 0, dsb.)

Ganti dengan isi asli dari repo deployment_project_used_car.
"""


def validate_input(data: dict) -> list[str]:
    """Return list pesan error; list kosong = valid."""
    errors = []

    if data.get("Year") and data["Year"] > 2022:
        errors.append("Year tidak boleh lebih dari 2022.")

    if data.get("Mileage") is not None and data["Mileage"] < 0:
        errors.append("Mileage tidak boleh negatif.")

    return errors
