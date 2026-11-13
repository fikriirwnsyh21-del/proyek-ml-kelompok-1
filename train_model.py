"""
Script Training & Evaluasi Supervised Learning (Klasifikasi Risiko Diabetes)
Dataset: diabetes_012_health_indicators_BRFSS2015.csv
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "diabetes_012_health_indicators_BRFSS2015.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "model_kelompok_1.joblib"

NUMERIC_FEATURES = [
    "BMI",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "Age",
    "Education",
    "Income",
]

CATEGORICAL_FEATURES = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "DiffWalk",
    "Sex",
]

TARGET_COL = "Diabetes_012"
CLASS_NAMES = ["Tidak Diabetes (0)", "Prediabetes (1)", "Diabetes (2)"]


def load_and_clean_data(csv_path: Path) -> pd.DataFrame:
    """Langkah 1: Eksplorasi & cleaning dataset."""
    df = pd.read_csv(csv_path)
    print(f"Ukuran awal dataset: {df.shape[0]} baris, {df.shape[1]} kolom")
    print(f"Total missing values: {int(df.isnull().sum().sum())}")

    dup_count = int(df.duplicated().sum())
    print(f"Total baris duplikat: {dup_count}")
    if dup_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Ukuran setelah hapus duplikat: {df.shape[0]} baris")

    print("\nDistribusi kelas target (Diabetes_012):")
    dist = df[TARGET_COL].value_counts().sort_index()
    for cls, cnt in dist.items():
        pct = (cnt / len(df)) * 100
        print(f"  Kelas {int(cls)}: {cnt} ({pct:.2f}%)")

    print("\nDeteksi outlier (metode IQR) pada fitur kontinu/hitungan:")
    for col in ["BMI", "MentHlth", "PhysHlth"]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
        print(f"  {col}: {outliers} outlier (batas IQR [{lower:.1f}, {upper:.1f}])")

    return df


def build_preprocessor() -> ColumnTransformer:
    """Langkah 2: ColumnTransformer untuk scaling numerik & encoding kategorikal."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(drop="if_binary", handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def evaluate_on_test(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluasi metrik klasifikasi pada Test Set."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n=== Evaluasi Test Set: {name} ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_weighted": f1,
    }


def main() -> None:
    df = load_and_clean_data(DATA_PATH)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nSplit data -> Train: {len(X_train)} baris, Test: {len(X_test)} baris")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, class_weight="balanced", random_state=42
        ),
        "MLP Neural Network": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=30,
            early_stopping=True,
            random_state=42,
        ),
    }

    print("\n=== Langkah 3: Stratified 5-Fold Cross Validation ===")
    results = []
    for name, clf in candidates.items():
        pipe = Pipeline(steps=[("preprocessor", build_preprocessor()), ("classifier", clf)])
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=-1)
        print(f"{name:22s} | CV F1-Weighted: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        pipe.fit(X_train, y_train)
        test_metrics = evaluate_on_test(name, pipe, X_test, y_test)
        test_metrics["cv_f1_mean"] = float(cv_scores.mean())
        results.append(test_metrics)

    # Hyperparameter tuning pada Decision Tree menggunakan GridSearchCV
    print("\n=== Hyperparameter Tuning (GridSearchCV - Decision Tree) ===")
    dt_pipe = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                DecisionTreeClassifier(class_weight="balanced", random_state=42),
            ),
        ]
    )
    param_grid = {
        "classifier__max_depth": [8, 10, 12],
        "classifier__min_samples_leaf": [10, 25, 50],
        "classifier__criterion": ["gini", "entropy"],
    }
    grid = GridSearchCV(
        dt_pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    print(f"Best Params: {grid.best_params_}")
    print(f"Best CV F1-Weighted: {grid.best_score_:.4f}")

    best_pipeline = grid.best_estimator_
    tuned_metrics = evaluate_on_test("Decision Tree (Tuned)", best_pipeline, X_test, y_test)
    tuned_metrics["cv_f1_mean"] = float(grid.best_score_)
    results.append(tuned_metrics)

    print("\n=== Ringkasan Perbandingan Model ===")
    summary_df = pd.DataFrame(results).sort_values(by="f1_weighted", ascending=False)
    print(summary_df.to_string(index=False))

    # Langkah 4: Ekspor model final terbaik
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_pipeline.fit(X_train, y_train)
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"\nModel final berhasil disimpan ke: {MODEL_PATH}")

    # Verifikasi load ulang
    loaded_model = joblib.load(MODEL_PATH)
    sample_pred = loaded_model.predict(X_test.iloc[:1])
    sample_prob = loaded_model.predict_proba(X_test.iloc[:1])
    print(f"Verifikasi load model -> Prediksi sampel: {sample_pred[0]}, Probabilitas: {sample_prob[0]}")


if __name__ == "__main__":
    main()
