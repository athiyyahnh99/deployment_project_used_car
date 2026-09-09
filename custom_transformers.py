"""
custom_transformers.py

WAJIB ada dan didefinisikan SEBELUM joblib.load() dipanggil, karena
model CatBoost yang di-pickle menyimpan referensi ke class ini.

Berisi:
- CarAgeTransformer (reference_year=2021)
- FrequencyEncoder (Make, Type, Region)

Ganti dengan isi asli dari repo deployment_project_used_car.
"""

from sklearn.base import BaseEstimator, TransformerMixin


class CarAgeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, reference_year: int = 2021):
        self.reference_year = reference_year

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        raise NotImplementedError("Ganti dengan implementasi asli.")


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        raise NotImplementedError("Ganti dengan implementasi asli.")
