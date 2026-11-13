"""
Pengujian End-to-End Backend API (12 Test Cases untuk Bab IV Laporan)
"""

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

VALID_BASE = {
    "BMI": 25.0,
    "GenHlth": 2,
    "MentHlth": 0,
    "PhysHlth": 0,
    "Age": 6,
    "Education": 5,
    "Income": 6,
    "HighBP": 0,
    "HighChol": 0,
    "CholCheck": 1,
    "Smoker": 0,
    "Stroke": 0,
    "HeartDiseaseorAttack": 0,
    "PhysActivity": 1,
    "Fruits": 1,
    "Veggies": 1,
    "HvyAlcoholConsump": 0,
    "AnyHealthcare": 1,
    "NoDocbcCost": 0,
    "DiffWalk": 0,
    "Sex": 0,
}


def run_tests() -> None:
    cases = [
        ("TC01: GET /health", "GET", "/health", None, 200),
        ("TC02: GET / (Frontend HTML)", "GET", "/", None, 200),
        ("TC03: POST /predict (Valid - Risiko Rendah)", "POST", "/predict", VALID_BASE, 200),
        (
            "TC04: POST /predict (Valid - Risiko Tinggi)",
            "POST",
            "/predict",
            {
                **VALID_BASE,
                "BMI": 40.0,
                "GenHlth": 5,
                "PhysHlth": 20,
                "Age": 11,
                "HighBP": 1,
                "HighChol": 1,
                "HeartDiseaseorAttack": 1,
                "DiffWalk": 1,
                "PhysActivity": 0,
            },
            200,
        ),
        (
            "TC05: POST /predict (Valid - Batas Minimum BMI=10)",
            "POST",
            "/predict",
            {**VALID_BASE, "BMI": 10.0, "GenHlth": 1, "Age": 1},
            200,
        ),
        (
            "TC06: POST /predict (Valid - Batas Maksimum BMI=100)",
            "POST",
            "/predict",
            {**VALID_BASE, "BMI": 100.0, "GenHlth": 5, "Age": 13},
            200,
        ),
        (
            "TC07: POST /predict (Out-of-Bound BMI=-5)",
            "POST",
            "/predict",
            {**VALID_BASE, "BMI": -5.0},
            422,
        ),
        (
            "TC08: POST /predict (Out-of-Bound BMI=150)",
            "POST",
            "/predict",
            {**VALID_BASE, "BMI": 150.0},
            422,
        ),
        (
            "TC09: POST /predict (Out-of-Bound GenHlth=6)",
            "POST",
            "/predict",
            {**VALID_BASE, "GenHlth": 6},
            422,
        ),
        (
            "TC10: POST /predict (Out-of-Bound HighBP=2)",
            "POST",
            "/predict",
            {**VALID_BASE, "HighBP": 2},
            422,
        ),
        (
            "TC11: POST /predict (Missing Field - tanpa BMI)",
            "POST",
            "/predict",
            {k: v for k, v in VALID_BASE.items() if k != "BMI"},
            422,
        ),
        (
            "TC12: POST /predict (Invalid Type - BMI string)",
            "POST",
            "/predict",
            {**VALID_BASE, "BMI": "bukan-angka"},
            422,
        ),
    ]

    passed = 0
    for name, method, path, payload, expected_status in cases:
        if method == "GET":
            resp = client.get(path)
        else:
            resp = client.post(path, json=payload)

        ok = resp.status_code == expected_status
        status_mark = "PASS" if ok else "FAIL"
        if ok:
            passed += 1

        detail = ""
        if path == "/predict" and resp.status_code == 200:
            body = resp.json()
            detail = f" -> {body['prediction_label']} | {body['probabilities_percent']}"
        elif path == "/health" and resp.status_code == 200:
            detail = f" -> {resp.json()}"

        print(f"[{status_mark}] {name} | HTTP {resp.status_code} (expected {expected_status}){detail}")
        assert ok, f"{name} failed: got {resp.status_code}, body={resp.text}"

    print(f"\nRingkasan: {passed}/{len(cases)} test cases PASSED.")


if __name__ == "__main__":
    run_tests()
