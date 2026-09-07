# Used Car Price Prediction — Streamlit App

Struktur mengikuti pola `deployment_project_travel_insurance`: satu file
`streamlit_app.py` di root dengan `st.tabs()` (Single Prediction / Batch
Upload CSV), plus package `app/` untuk konstanta dan helper.

## Struktur folder

```
car_price_app/
├── streamlit_app.py                    # entry point, tabs Single Prediction & Batch Upload
├── requirements.txt
├── .gitignore
└── app/
    ├── __init__.py
    ├── constants.py                    # daftar Make/Type/Region dkk dari data training
    ├── models/
    │   └── final_model_catboost.joblib # <-- TAMBAHKAN FILE INI SENDIRI
    └── utils/
        ├── __init__.py
        ├── custom_transformers.py      # WAJIB (CarAgeTransformer, FrequencyEncoder)
        ├── model_helper.py             # load_model() + predict_price()
        └── validation.py               # validasi input sebelum prediksi
```

## Langkah menjalankan

1. Export model dari notebook (kalau belum): jalankan cell
   `joblib.dump(final_model, 'final_model_catboost.joblib')`, lalu copy file
   hasilnya ke `app/models/final_model_catboost.joblib`.
2. `pip install -r requirements.txt`
3. `streamlit run streamlit_app.py`

## Upload ke GitHub

1. `final_model_catboost.joblib` — cek dulu ukurannya (`ls -lh`). Kalau di
   bawah ~100MB masih bisa langsung di-commit; kalau lebih besar, pakai
   [Git LFS](https://git-lfs.com/) (`git lfs track "*.joblib"`).
2. Dataset CSV mentah (`UsedCarsSA_*.csv`) sebaiknya ditaruh di repo utama
   (folder terpisah, misal `data/`), bukan di dalam folder app ini, biar tidak
   duplikat dengan yang dipakai notebook.
3. `.gitignore` sudah disiapkan untuk `__pycache__/`, venv, dll.
4. Struktur push-nya bisa dua cara:
   - **Repo terpisah** khusus deployment (kayak repo referensi) → langsung
     `git init` di folder `car_price_app/` ini.
   - **Subfolder di repo utama** (gabung sama notebook + dataset, ngikutin
     pola `[TeamName]_JC_DS_[Batch]_FinalProject`) → taruh folder ini sebagai
     `deployment/` atau `streamlit/` di root repo.

```bash
cd car_price_app
git init
git add .
git commit -m "Add Streamlit deployment for used car price prediction"
git branch -M main
git remote add origin https://github.com/<username>/<repo-name>.git
git push -u origin main
```

## Catatan

- Tema dark + aksen merah di screenshot referensi itu **bukan custom CSS** —
  itu tema dark bawaan Streamlit (`primaryColor` default `#FF4B4B`), aktif
  otomatis kalau browser/OS user dalam mode dark. App ini tidak butuh
  konfigurasi tambahan untuk itu.
- 11 kolom fitur mentah yang wajib ada: `Make, Type, Year, Origin, Color,
  Options, Engine_Size, Fuel_Type, Gear_Type, Mileage, Region` — persis
  seperti `X_train` sebelum masuk pipeline di notebook.
- Tab "Batch Upload (CSV)" menerima file dengan 11 kolom di atas, menambahkan
  kolom `predicted_price_sar`, dan bisa langsung di-download hasilnya.
