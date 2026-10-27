"""Custom transformer untuk pipeline prediksi harga mobil bekas Syarah.com."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

REFERENCE_YEAR = 2021


def add_car_age(X):
    """Mengubah Year menjadi Car_Age dengan acuan tahun pengumpulan data.

    Args:
        X: DataFrame yang memiliki kolom Year.

    Returns:
        DataFrame dengan kolom Year diganti menjadi Car_Age (minimal 0).
    """
    X = X.copy()
    X['Year'] = (REFERENCE_YEAR - X['Year']).clip(lower=0)
    return X.rename(columns={'Year': 'Car_Age'})


def car_age_feature_names(transformer, input_features):
    """Menyesuaikan nama kolom output agar Year tercatat sebagai Car_Age."""
    return np.array(['Car_Age' if col == 'Year' else col for col in input_features])


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Mengisi NaN dengan median bertingkat per kelompok yang dihitung dari data train.

    Args:
        target_col: Kolom yang akan diimputasi.
        group_levels: Daftar kelompok dari yang paling spesifik ke paling umum,
            misalnya [['Make', 'Type'], ['Make']]. Jika kelompok tidak ditemukan,
            digunakan level berikutnya, lalu median global.
    """

    def __init__(self, target_col, group_levels):
        self.target_col = target_col
        self.group_levels = group_levels

    def fit(self, X, y=None):
        self.medians_ = [
            X.groupby(g)[self.target_col].median().rename('_median').reset_index()
            for g in self.group_levels
        ]
        self.global_median_ = X[self.target_col].median()
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X):
        X = X.copy()
        for group, median in zip(self.group_levels, self.medians_):
            mask = X[self.target_col].isna()
            if not mask.any():
                break
            # merge mempertahankan urutan baris, sehingga hasilnya sejajar dengan mask
            fill = X.loc[mask, group].merge(median, on=group, how='left')['_median']
            X.loc[mask, self.target_col] = fill.to_numpy()
        X[self.target_col] = X[self.target_col].fillna(self.global_median_)
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_
