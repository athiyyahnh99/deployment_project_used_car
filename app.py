"""Aplikasi Streamlit estimasi harga mobil bekas (Syarah.com, 2021).

Model: XGBoost tuned (TransformedTargetRegressor log1p/expm1) yang disimpan
sebagai dictionary joblib di `models/used_car_price_model.pkl`.
Rentang harga = estimasi ± MAPE data test, konsisten dengan metrik utama proyek.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "used_car_price_model.pkl"

# car_preprocessing.py harus bisa di-import dengan nama yang sama seperti saat
# model disimpan, karena pickle hanya menyimpan referensi ke class/fungsinya.
sys.path.insert(0, str(APP_DIR))
import car_preprocessing  # noqa: E402,F401

REQUIRED_KEYS = {"model", "features", "categories", "make_to_type", "year_range", "test_metrics"}
OPTIONS_ORDER = ["Standard", "Semi Full", "Full"]


# ----------------------------------------------------------------------------
# Data & model
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifact(path: Path = MODEL_PATH) -> dict:
    """Memuat artefak model dan memvalidasi isinya.

    Args:
        path: Lokasi file .pkl hasil joblib.dump di notebook modeling.

    Returns:
        Dictionary berisi model, daftar fitur, kategori, dan metrik test.

    Raises:
        FileNotFoundError: Jika file model tidak ditemukan.
        KeyError: Jika ada key wajib yang hilang dari artefak.
    """
    if not path.exists():
        raise FileNotFoundError(f"File model tidak ditemukan: {path}")

    artifact = joblib.load(path)
    missing = REQUIRED_KEYS - set(artifact)
    if missing:
        raise KeyError(f"Key berikut tidak ada di artefak: {sorted(missing)}")
    assert "MAPE" in artifact["test_metrics"], "test_metrics harus berisi MAPE"
    assert 0 < artifact["test_metrics"]["MAPE"] < 1, "MAPE harus berupa pecahan (mis. 0.183)"
    return artifact


def build_input(values: dict, features: list[str]) -> pd.DataFrame:
    """Menyusun input pengguna menjadi DataFrame satu baris sesuai urutan fitur model.

    Args:
        values: Nilai setiap fitur dari form. Mileage/Engine_Size boleh NaN
            (akan diimputasi oleh GroupMedianImputer di pipeline).
        features: Urutan kolom yang dipakai saat training.

    Returns:
        DataFrame berukuran (1, len(features)).
    """
    df = pd.DataFrame([values])[features]
    for col in ["Year", "Mileage", "Engine_Size"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    assert df.shape == (1, len(features)), f"Shape input tidak sesuai: {df.shape}"
    assert df["Year"].notna().all(), "Year wajib diisi"
    return df


def predict_with_range(model, X: pd.DataFrame, mape: float) -> tuple[float, float, float]:
    """Menghasilkan estimasi harga beserta rentang ± MAPE.

    Args:
        model: Pipeline terlatih (output sudah dalam SAR).
        X: Input satu baris dari `build_input`.
        mape: MAPE data test dalam bentuk pecahan.

    Returns:
        Tuple (estimasi, batas bawah, batas atas) dalam SAR.
    """
    pred = float(model.predict(X)[0])
    assert np.isfinite(pred) and pred > 0, f"Prediksi tidak valid: {pred}"
    return pred, pred * (1 - mape), pred * (1 + mape)


def ordered(values: list[str], preferred: list[str]) -> list[str]:
    """Mengurutkan kategori sesuai urutan logis jika tersedia, sisanya di belakang."""
    head = [v for v in preferred if v in values]
    return head + [v for v in values if v not in head]


# ----------------------------------------------------------------------------
# Tampilan
# ----------------------------------------------------------------------------
def render_about(mape: float) -> None:
    """Menampilkan penjelasan aplikasi yang bisa dibuka-tutup."""
    with st.expander("ℹ️ Tentang aplikasi ini", expanded=True):
        st.markdown(
            f"""
**Apa fungsi aplikasi ini?**

Aplikasi ini memperkirakan harga pasar mobil bekas di Arab Saudi berdasarkan
spesifikasinya. Model dilatih dari data listing Syarah.com tahun 2021 dan
dirancang sebagai acuan harga awal sebelum mobil diinspeksi.

**Cara pakai**

1. Pilih merek, lalu model mobil (pilihan model menyesuaikan merek).
2. Isi tahun, kelengkapan, transmisi, bahan bakar, asal, warna, dan wilayah.
3. Isi jarak tempuh dan kapasitas mesin. Kalau tidak tahu, centang
   "tidak diketahui" dan model akan memakai nilai tengah dari mobil sejenis.
4. Klik **Estimasi Harga**.

**Cara membaca hasil**

Hasil ditampilkan sebagai rentang harga, yaitu estimasi ± {mape:.1%}.
Angka {mape:.1%} adalah rata-rata selisih prediksi model dengan harga asli
pada data uji (MAPE). Harga asli sebagian mobil bisa berada di luar rentang ini.

**Keterbatasan**

- Model tidak mengetahui kondisi fisik, riwayat servis, atau riwayat kecelakaan mobil.
- Estimasi untuk model mobil yang jarang muncul di data kurang akurat.
- Harga mengacu pada kondisi pasar tahun 2021.
"""
        )
        
def render_form(artifact: dict) -> dict:
    """Menampilkan input fitur mobil dan mengembalikan nilainya."""
    cats = artifact["categories"]
    year_min, year_max = artifact["year_range"]

    st.subheader("Data Mobil")
    col1, col2 = st.columns(2)

    with col1:
        make = st.selectbox("Merek (Make)", sorted(artifact["make_to_type"]))
        car_type = st.selectbox("Model (Type)", artifact["make_to_type"][make])
        year = st.number_input("Tahun", min_value=year_min, max_value=year_max,
                               value=max(year_min, year_max - 5), step=1)
        options = st.selectbox("Kelengkapan (Options)", ordered(cats["Options"], OPTIONS_ORDER))
        gear = st.selectbox("Transmisi (Gear_Type)", cats["Gear_Type"])
        fuel = st.selectbox("Bahan Bakar (Fuel_Type)", cats["Fuel_Type"])

    with col2:
        origin = st.selectbox("Asal (Origin)", cats["Origin"])
        color = st.selectbox("Warna (Color)", cats["Color"])
        region = st.selectbox("Wilayah (Region)", cats["Region"])

        mileage_unknown = st.checkbox("Jarak tempuh tidak diketahui")
        mileage = st.number_input("Jarak Tempuh (km)", min_value=0, max_value=1_000_000,
                                  value=100_000, step=1_000, disabled=mileage_unknown)

        engine_unknown = st.checkbox("Kapasitas mesin tidak diketahui")
        engine = st.number_input("Kapasitas Mesin (L)", min_value=1.0, max_value=9.0,
                                 value=2.5, step=0.1, disabled=engine_unknown)

    return {
        "Make": make, "Type": car_type, "Year": int(year), "Origin": origin,
        "Color": color, "Options": options, "Engine_Size": np.nan if engine_unknown else engine,
        "Fuel_Type": fuel, "Gear_Type": gear, "Mileage": np.nan if mileage_unknown else mileage,
        "Region": region,
    }


def render_result(pred: float, lower: float, upper: float, mape: float) -> None:
    """Menampilkan rentang harga sebagai hasil utama, estimasi titik sebagai acuan."""
    st.subheader("Hasil Estimasi")
    st.metric("Perkiraan Harga Pasar", f"{lower:,.0f} – {upper:,.0f} SAR")
    st.caption(
        f"Titik tengah estimasi: {pred:,.0f} SAR. "
        f"Rentang = estimasi ± {mape:.1%} (MAPE model pada data test)."
    )

def validate_csv(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Memvalidasi CSV upload dan menyiapkan kolom fitur untuk prediksi.

    Args:
        df: Data hasil upload. Kolom tambahan (mis. Price, Link) dibiarkan.
        artifact: Artefak model berisi daftar fitur dan kategori.

    Returns:
        DataFrame fitur (urutan sesuai training) untuk baris yang valid.

    Raises:
        ValueError: Jika ada kolom fitur yang hilang atau tidak ada baris valid.
    """
    features = artifact["features"]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom berikut tidak ada di CSV: {missing}")

    X = df[features].copy()
    for col in ["Year", "Mileage", "Engine_Size"]:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X[X["Year"].notna()]
    if X.empty:
        raise ValueError("Tidak ada baris dengan Year yang valid.")
    assert list(X.columns) == features, "Urutan kolom tidak sesuai"
    return X


def unseen_categories(X: pd.DataFrame, categories: dict) -> dict:
    """Menghitung nilai kategori yang tidak pernah muncul di data train."""
    result = {}
    for col, known in categories.items():
        n = int((~X[col].isin(known) & X[col].notna()).sum())
        if n:
            result[col] = n
    return result


def render_batch(artifact: dict) -> None:
    """Menampilkan upload CSV, prediksi massal, dan tombol download hasil."""
    features = artifact["features"]
    mape = artifact["test_metrics"]["MAPE"]

    st.write("Upload file CSV berisi kolom: " + ", ".join(f"`{c}`" for c in features))
    st.download_button(
        "Download template CSV",
        pd.DataFrame(columns=features).to_csv(index=False),
        file_name="template_input.csv",
        mime="text/csv",
    )

    file = st.file_uploader("Upload CSV", type="csv")
    if file is None:
        return

    df = pd.read_csv(file)
    try:
        X = validate_csv(df, artifact)
    except ValueError as err:
        st.error(str(err))
        return

    skipped = len(df) - len(X)
    if skipped:
        st.warning(f"{skipped} baris dilewati karena Year kosong atau tidak valid.")
    unseen = unseen_categories(X, artifact["categories"])
    if unseen:
        st.warning(f"Ada nilai kategori yang tidak dikenal model (jumlah baris): {unseen}. "
                   "Estimasi untuk baris tersebut kurang akurat.")

    pred = artifact["model"].predict(X)
    result = df.loc[X.index].copy()
    result["Estimasi"] = pred.round(0)
    result["Batas_Bawah"] = (pred * (1 - mape)).round(0)
    result["Batas_Atas"] = (pred * (1 + mape)).round(0)
    assert result["Estimasi"].gt(0).all(), "Ada prediksi tidak valid"

    st.success(f"{len(result):,} mobil berhasil diestimasi.")
    st.dataframe(result, use_container_width=True)
    st.download_button(
        "Download hasil estimasi",
        result.to_csv(index=False),
        file_name="hasil_estimasi.csv",
        mime="text/csv",
        type="primary",
    )


def main() -> None:
    st.set_page_config(page_title="Estimasi Harga Mobil Bekas", page_icon="🚗", layout="centered")
    st.title("🚗 Estimasi Harga Mobil Bekas")
    st.write("Estimasi harga pasar mobil bekas di Arab Saudi berdasarkan data listing Syarah.com tahun 2021.")

    try:
        artifact = load_artifact()
    except (FileNotFoundError, KeyError, AssertionError) as err:
        st.error(f"Model gagal dimuat: {err}")
        st.stop()
        
    render_about(artifact["test_metrics"]["MAPE"])
    tab_single, tab_batch = st.tabs(["Satu Mobil", "Upload CSV"])

    with tab_single:
        values = render_form(artifact)
        if st.button("Estimasi Harga", type="primary", use_container_width=True):
            X = build_input(values, artifact["features"])
            mape = artifact["test_metrics"]["MAPE"]
            render_result(*predict_with_range(artifact["model"], X, mape), mape=mape)

    with tab_batch:
        render_batch(artifact)


if __name__ == "__main__":
    main()
