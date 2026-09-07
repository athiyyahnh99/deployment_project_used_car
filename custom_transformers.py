"""
custom_transformers.py

Custom transformer classes untuk pipeline prediksi harga mobil bekas Saudi Arabia.
File ini WAJIB tersedia (di folder yang sama atau ter-install sebagai module)
setiap kali pipeline (`final_model_catboost.joblib`) di-load ulang dengan joblib,
baik di notebook lain, script terpisah, maupun API/aplikasi deployment.

Cara pakai:
    from custom_transformers import CarAgeTransformer, FrequencyEncoder

    import joblib
    pipeline = joblib.load("final_model_catboost.joblib")
    pred_log = pipeline.predict(new_data)
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CarAgeTransformer(BaseEstimator, TransformerMixin):
    """
    Mengubah kolom 'Year' menjadi 'car_age' sebagai bagian dari pipeline,
    supaya model bisa menerima data mentah (dengan kolom 'Year') langsung,
    tanpa perlu feature engineering manual di luar pipeline.

    Parameters
    ----------
    reference_year : int, default=2021
        Tahun acuan untuk menghitung usia mobil (car_age = reference_year - Year).
        Nilai negatif (Year > reference_year) di-clip menjadi 0.
    """

    def __init__(self, reference_year=2021):
        self.reference_year = reference_year

    def fit(self, X, y=None):
        # Stateless — tidak ada apapun yang perlu dipelajari dari data.
        return self

    def transform(self, X):
        X = X.copy()
        X['car_age'] = (self.reference_year - X['Year']).clip(lower=0)
        X = X.drop(columns=['Year'])
        return X


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Encode kolom kategorikal berdasarkan frekuensi kemunculan (proporsi 0-1)
    di data training. Kategori yang tidak muncul saat training (hanya ada
    saat predict) diberi nilai 0.

    Dipakai untuk fitur berkardinalitas tinggi: Make, Type, Region.
    """

    def fit(self, X, y=None):
        self.freq_maps_ = []
        for i in range(X.shape[1]):
            values, counts = np.unique(X[:, i], return_counts=True)
            freq = dict(zip(values, counts / counts.sum()))
            self.freq_maps_.append(freq)
        return self

    def transform(self, X):
        X_out = np.zeros_like(X, dtype=float)
        for i, freq_map in enumerate(self.freq_maps_):
            X_out[:, i] = [freq_map.get(val, 0.0) for val in X[:, i]]
        return X_out

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array([f"freq_{name}" for name in input_features])
        return np.array([f"freq_{i}" for i in range(len(self.freq_maps_))])
