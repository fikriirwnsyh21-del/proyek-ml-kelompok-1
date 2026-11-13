# Diabetes Early Risk Screening Web App

Aplikasi web Machine Learning untuk screening awal risiko penyakit diabetes berbasis dataset **BRFSS 2015 Health Indicators** (Tugas Besar Praktikum Pembelajaran Mesin).

## Struktur Proyek
```text
proyek-ml-kelompok 1/
├── app.py                 # REST API FastAPI + validasi Pydantic
├── requirements.txt       # Daftar dependensi library Python
├── test_api.py            # Skrip otomatis 12 end-to-end API test cases
├── model/
│   └── model_kelompok_1.joblib   # Pipeline model terbaik scikit-learn
└── static/
    ├── index.html         # Tampilan antarmuka formulir interaktif
    ├── style.css          # Desain responsif (Desktop & Mobile)
    └── app.js             # Logika Fetch API asinkron (tanpa page reload)
```

## Cara Menjalankan Proyek di Lokal

1. **Clone repository ini**:
   ```bash
   git clone <URL_REPOSITORY_GITHUB_ANDA>
   cd "proyek-ml-kelompok 1"
   ```

2. **Buat dan aktifkan Virtual Environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\activate
   ```

3. **Install dependensi**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan pengujian API**:
   ```bash
   python test_api.py
   ```

5. **Jalankan Web Server**:
   ```bash
   uvicorn app:app --reload
   ```
   Buka browser di:
   - Web App: `http://127.0.0.1:8000`
   - Dokumentasi Swagger API: `http://127.0.0.1:8000/docs`
