from pathlib import Path
import time

import cv2
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedGroupKFold,
    cross_validate,
    cross_val_predict,
    GridSearchCV,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

from features import (
    preprocess_image,
    extract_hog,
    extract_hsv,
)


# ============================================================
# RUTAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset" / "raw"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

CACHE_FILE = RESULTS_DIR / "features_hog_hsv.npz"


# ============================================================
# EXTRAER SCENE ID
# ============================================================

def get_scene_id(filename):

    stem = Path(filename).stem

    parts = stem.split("__")

    if len(parts) < 2:
        return "UNKNOWN"

    return parts[1]


# ============================================================
# CARGAR / EXTRAER FEATURES
# ============================================================

def load_features():

    # --------------------------------------------------------
    # Si ya se calcularon antes, cargarlas
    # --------------------------------------------------------

    if CACHE_FILE.exists():

        print("Cargando características desde caché...")

        data = np.load(
            CACHE_FILE,
            allow_pickle=True
        )

        return (
            data["X"],
            data["y"],
            data["groups"],
            data["filenames"]
        )

    # --------------------------------------------------------
    # Si no existen, calcularlas
    # --------------------------------------------------------

    files = sorted(
        DATASET_DIR.glob("*.png")
    )

    if not files:
        raise RuntimeError(
            f"No hay imágenes en:\n{DATASET_DIR}"
        )

    X = []
    y = []
    groups = []
    filenames = []

    print("=" * 70)
    print("EXTRACCIÓN HOG + HSV")
    print("=" * 70)

    for i, path in enumerate(files):

        label = int(
            path.name.split("__")[0]
        )

        scene_id = get_scene_id(
            path.name
        )

        image = cv2.imread(
            str(path)
        )

        if image is None:
            continue

        image = preprocess_image(
            image
        )

        hog_features = extract_hog(
            image
        )

        hsv_features = extract_hsv(
            image
        )

        features = np.concatenate([
            hog_features,
            hsv_features
        ])

        X.append(features)
        y.append(label)
        groups.append(scene_id)
        filenames.append(path.name)

        if (i + 1) % 500 == 0:

            print(
                f"Procesadas: "
                f"{i + 1}/{len(files)}"
            )

    X = np.array(
        X,
        dtype=np.float32
    )

    y = np.array(
        y,
        dtype=np.int32
    )

    groups = np.array(
        groups
    )

    filenames = np.array(
        filenames
    )

    # Guardar caché
    np.savez_compressed(
        CACHE_FILE,
        X=X,
        y=y,
        groups=groups,
        filenames=filenames
    )

    print(
        f"\nCaracterísticas guardadas en:\n"
        f"{CACHE_FILE}"
    )

    return (
        X,
        y,
        groups,
        filenames
    )


# ============================================================
# MODELO BASE
# ============================================================

def build_model(
    C=10,
    gamma="scale"
):

    return Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            SVC(
                kernel="rbf",
                C=C,
                gamma=gamma,
                class_weight="balanced"
            )
        )
    ])


# ============================================================
# VALIDACIÓN CRUZADA
# ============================================================

def evaluate_cv(
    name,
    model,
    X,
    y,
    cv,
    groups=None
):

    print("\n")
    print("=" * 70)
    print(name)
    print("=" * 70)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "balanced_accuracy":
            "balanced_accuracy",
    }

    start = time.time()

    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        groups=groups,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )

    elapsed = time.time() - start

    summary = {}

    for metric in scoring.keys():

        values = results[
            f"test_{metric}"
        ]

        mean = values.mean()
        std = values.std()

        summary[metric] = mean
        summary[f"{metric}_std"] = std

        print(
            f"{metric:18s}: "
            f"{mean * 100:.3f}% "
            f"± {std * 100:.3f}%"
        )

    print(
        f"\nTiempo: {elapsed:.2f} s"
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    X, y, groups, filenames = (
        load_features()
    )

    print("\n")
    print("=" * 70)
    print("DATASET")
    print("=" * 70)

    print(
        f"Imágenes: {len(y)}"
    )

    print(
        f"Features por imagen: "
        f"{X.shape[1]}"
    )

    print(
        f"No ship: {(y == 0).sum()}"
    )

    print(
        f"Ship: {(y == 1).sum()}"
    )

    print(
        f"Scene IDs: "
        f"{len(np.unique(groups))}"
    )

    # ========================================================
    # MODELO ACTUAL
    # ========================================================

    baseline_model = build_model(
        C=10,
        gamma="scale"
    )

    # ========================================================
    # CV NORMAL
    # ========================================================

    normal_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    normal_results = evaluate_cv(
        "VALIDACIÓN CRUZADA ESTRATIFICADA",
        baseline_model,
        X,
        y,
        normal_cv
    )

    # ========================================================
    # CV AGRUPADA POR ESCENA
    # ========================================================

    group_cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    group_results = evaluate_cv(
        "VALIDACIÓN CRUZADA AGRUPADA POR SCENE ID",
        baseline_model,
        X,
        y,
        group_cv,
        groups=groups
    )

    # ========================================================
    # GUARDAR COMPARACIÓN
    # ========================================================

    comparison = pd.DataFrame([
        {
            "Validation":
                "StratifiedKFold",
            **normal_results
        },
        {
            "Validation":
                "StratifiedGroupKFold",
            **group_results
        }
    ])

    comparison.to_csv(
        RESULTS_DIR /
        "cross_validation_comparison.csv",
        index=False
    )

    # ========================================================
    # GRID SEARCH
    # ========================================================

    print("\n")
    print("=" * 70)
    print("GRID SEARCH")
    print("=" * 70)

    model = build_model()

    param_grid = {

        "classifier__C": [
            1,
            3,
            10,
            30,
            100
        ],

        "classifier__gamma": [
            "scale",
            0.0001,
            0.001,
            0.01
        ]
    }

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "balanced_accuracy":
            "balanced_accuracy"
    }

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        refit="accuracy",
        cv=group_cv,
        n_jobs=-1,
        verbose=2,
        return_train_score=False
    )

    start = time.time()

    grid.fit(
        X,
        y,
        groups=groups
    )

    elapsed = time.time() - start

    print("\n")
    print("=" * 70)
    print("MEJORES HIPERPARÁMETROS")
    print("=" * 70)

    print(
        grid.best_params_
    )

    print(
        f"\nMejor accuracy CV: "
        f"{grid.best_score_ * 100:.3f}%"
    )

    print(
        f"\nTiempo GridSearch: "
        f"{elapsed:.2f} s"
    )

    # ========================================================
    # GUARDAR TODOS LOS RESULTADOS DEL GRID
    # ========================================================

    grid_results = pd.DataFrame(
        grid.cv_results_
    )

    columns_to_save = [
        "param_classifier__C",
        "param_classifier__gamma",
        "mean_test_accuracy",
        "std_test_accuracy",
        "mean_test_precision",
        "mean_test_recall",
        "mean_test_f1",
        "mean_test_balanced_accuracy",
        "mean_fit_time"
    ]

    grid_results[
        columns_to_save
    ].sort_values(
        by="mean_test_accuracy",
        ascending=False
    ).to_csv(
        RESULTS_DIR /
        "grid_search_results.csv",
        index=False
    )

    # ========================================================
    # PREDICCIONES OUT-OF-FOLD
    # ========================================================

    print("\n")
    print("=" * 70)
    print("EVALUACIÓN OUT-OF-FOLD")
    print("=" * 70)

    best_model = grid.best_estimator_

    predictions = cross_val_predict(
        best_model,
        X,
        y,
        cv=group_cv,
        groups=groups,
        n_jobs=-1
    )

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    balanced_acc = (
        balanced_accuracy_score(
            y,
            predictions
        )
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    print(
        f"Accuracy          : "
        f"{accuracy * 100:.3f}%"
    )

    print(
        f"Precision         : "
        f"{precision * 100:.3f}%"
    )

    print(
        f"Recall            : "
        f"{recall * 100:.3f}%"
    )

    print(
        f"F1                : "
        f"{f1 * 100:.3f}%"
    )

    print(
        f"Balanced Accuracy : "
        f"{balanced_acc * 100:.3f}%"
    )

    print(
        "\nMatriz de confusión:"
    )

    print(cm)

    # ========================================================
    # GUARDAR PREDICCIONES
    # ========================================================

    predictions_df = pd.DataFrame({
        "filename": filenames,
        "scene_id": groups,
        "real_label": y,
        "predicted_label":
            predictions
    })

    predictions_df.to_csv(
        RESULTS_DIR /
        "oof_predictions.csv",
        index=False
    )

    # ========================================================
    # ENTRENAR MODELO FINAL CON TODO EL DATASET
    # ========================================================

    print("\n")
    print("=" * 70)
    print("ENTRENANDO MODELO FINAL")
    print("=" * 70)

    best_model.fit(
        X,
        y
    )

    model_path = (
        MODELS_DIR /
        "ship_classifier_svm.joblib"
    )

    joblib.dump(
        best_model,
        model_path
    )

    print(
        f"\nModelo guardado en:\n"
        f"{model_path}"
    )

    # ========================================================
    # RESUMEN FINAL
    # ========================================================

    summary = pd.DataFrame([
        {
            "Accuracy":
                accuracy,
            "Precision":
                precision,
            "Recall":
                recall,
            "F1":
                f1,
            "Balanced_Accuracy":
                balanced_acc,
            "Best_C":
                grid.best_params_[
                    "classifier__C"
                ],
            "Best_gamma":
                grid.best_params_[
                    "classifier__gamma"
                ],
        }
    ])

    summary.to_csv(
        RESULTS_DIR /
        "optimized_model_summary.csv",
        index=False
    )

    print("\n")
    print("=" * 70)
    print("FASE 4 TERMINADA")
    print("=" * 70)


if __name__ == "__main__":
    main()