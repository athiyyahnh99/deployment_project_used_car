# Estimasi Harga Mobil Bekas — Syarah.com

Aplikasi Streamlit untuk mengestimasi harga pasar mobil bekas di Arab Saudi, berdasarkan model XGBoost yang dilatih dengan data listing Syarah.com tahun 2021. Hasil ditampilkan sebagai estimasi harga beserta rentang ± MAPE model pada data test.

## Struktur

```
.
├── app.py                         # Aplikasi Streamlit
├── car_preprocessing.py           # Custom transformer (add_car_age, GroupMedianImputer)
├── models/
│   └── used_car_price_model.pkl   # Artefak model (joblib)
├── requirements.txt
└── README.md
```

## Menjalankan secara lokal

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud

1. Push repo ini ke GitHub.
2. Buka share.streamlit.io → **Create app** → pilih repo, branch, dan `app.py`.
3. Di **Advanced settings**, pilih **Python 3.13** (sama dengan versi saat training).

## Input model

`Make`, `Type`, `Year`, `Origin`, `Color`, `Options`, `Engine_Size`, `Fuel_Type`, `Gear_Type`, `Mileage`, `Region`. `Mileage` dan `Engine_Size` boleh dikosongkan; nilainya diimputasi oleh pipeline (median per kelompok).

## Keterbatasan

Estimasi untuk model mobil yang jarang muncul di data kurang akurat karena keterbatasan jumlah data.
