"""
Backend REST API FastAPI untuk Screening Awal Risiko Diabetes
"""

from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model_kelompok_1.joblib"
STATIC_DIR = BASE_DIR / "static"

FEATURE_ORDER = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
]

CLASS_LABELS = {
    0: "Tidak Diabetes",
    1: "Prediabetes (Risiko Sedang)",
    2: "Diabetes (Risiko Tinggi)",
}

app = FastAPI(
    title="Diabetes Early Risk Screening API",
    description="Prototipe edukatif REST API FastAPI untuk prediksi risiko diabetes berbasis BRFSS 2015.",
    version="1.0.0",
)

model = None
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class DiabetesInput(BaseModel):
    BMI: float = Field(..., ge=10.0, le=100.0, description="Body Mass Index (10 - 100)")
    GenHlth: int = Field(..., ge=1, le=5, description="Kesehatan umum (1=Sangat Baik, 5=Buruk)")
    MentHlth: int = Field(..., ge=0, le=30, description="Hari kesehatan mental kurang baik (0 - 30)")
    PhysHlth: int = Field(..., ge=0, le=30, description="Hari kesehatan fisik kurang baik (0 - 30)")
    Age: int = Field(..., ge=1, le=13, description="Kategori usia BRFSS (1=18-24 ... 13=80+)")
    Education: int = Field(..., ge=1, le=6, description="Tingkat pendidikan (1 - 6)")
    Income: int = Field(..., ge=1, le=8, description="Tingkat pendapatan (1 - 8)")
    HighBP: int = Field(..., ge=0, le=1, description="Tekanan darah tinggi (0=Tidak, 1=Ya)")
    HighChol: int = Field(..., ge=0, le=1, description="Kolesterol tinggi (0=Tidak, 1=Ya)")
    CholCheck: int = Field(..., ge=0, le=1, description="Cek kolesterol 5 tahun terakhir (0=Tidak, 1=Ya)")
    Smoker: int = Field(..., ge=0, le=1, description="Perokok >= 100 batang seumur hidup (0=Tidak, 1=Ya)")
    Stroke: int = Field(..., ge=0, le=1, description="Riwayat stroke (0=Tidak, 1=Ya)")
    HeartDiseaseorAttack: int = Field(..., ge=0, le=1, description="Penyakit jantung koroner/serangan jantung (0=Tidak, 1=Ya)")
    PhysActivity: int = Field(..., ge=0, le=1, description="Aktivitas fisik 30 hari terakhir (0=Tidak, 1=Ya)")
    Fruits: int = Field(..., ge=0, le=1, description="Konsumsi buah >= 1x/hari (0=Tidak, 1=Ya)")
    Veggies: int = Field(..., ge=0, le=1, description="Konsumsi sayur >= 1x/hari (0=Tidak, 1=Ya)")
    HvyAlcoholConsump: int = Field(..., ge=0, le=1, description="Konsumsi alkohol berat (0=Tidak, 1=Ya)")
    AnyHealthcare: int = Field(..., ge=0, le=1, description="Memiliki jaminan kesehatan (0=Tidak, 1=Ya)")
    NoDocbcCost: int = Field(..., ge=0, le=1, description="Tidak ke dokter karena biaya (0=Tidak, 1=Ya)")
    DiffWalk: int = Field(..., ge=0, le=1, description="Kesulitan berjalan/naik tangga (0=Tidak, 1=Ya)")
    Sex: int = Field(..., ge=0, le=1, description="Jenis kelamin (0=Perempuan, 1=Laki-laki)")


@app.get("/")
def serve_frontend() -> FileResponse:
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html tidak ditemukan.")
    return FileResponse(str(index_file))


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH.name),
    }


@app.post("/predict")
def predict(payload: DiabetesInput) -> dict:
    global model
    if model is None:
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
        else:
            raise HTTPException(
                status_code=503,
                detail="Model belum tersedia. Jalankan train_model.py terlebih dahulu.",
            )

    input_dict = payload.model_dump()
    df_input = pd.DataFrame([input_dict], columns=FEATURE_ORDER)

    pred_class = int(model.predict(df_input)[0])
    probs = model.predict_proba(df_input)[0]

    probabilities = {
        CLASS_LABELS[idx]: round(float(prob) * 100, 2)
        for idx, prob in enumerate(probs)
    }

    return {
        "prediction_code": pred_class,
        "prediction_label": CLASS_LABELS.get(pred_class, f"Kelas {pred_class}"),
        "probabilities_percent": probabilities,
        "disclaimer": (
            "Sistem ini merupakan prototipe edukatif untuk keperluan akademik "
            "dan bukan penentu keputusan medis profesional."
        ),
    }
