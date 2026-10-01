from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from features import (
    preprocess_image,
    extract_hog,
    extract_hsv,
    extract_lbp
)


# ============================================================
# RUTAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset" / "raw"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# CARGA DEL DATASET
# ============================================================

def load_dataset():

    files = sorted(DATASET_DIR.glob("*.png"))

    if not files:
        raise RuntimeError(
            f"No se encontraron imágenes en:\n{DATASET_DIR}"
        )

    print("=" * 65)
    print("EXTRACCIÓN DE CARACTERÍSTICAS")
    print("=" * 65)

    X_rgb = []
    X_hog = []
    X_hog_hsv = []
    X_full = []

    y = []

    filenames = []

    for i, path in enumerate(files):

        # ----------------------------------------------------
        # Etiqueta desde el nombre
        # ----------------------------------------------------

        label = int(path.name.split("__")[0])

        # ----------------------------------------------------
        # Leer imagen
        # ----------------------------------------------------

        image = cv2.imread(str(path))

        if image is None:
            print(f"ERROR leyendo: {path.name}")
            continue

        image = preprocess_image(image)

        # ----------------------------------------------------
        # BASELINE RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        rgb_features = (
            rgb.astype(np.float32).flatten() / 255.0
        )

        # ----------------------------------------------------
        # HOG
        # ----------------------------------------------------

        hog_features = extract_hog(image)

        # ----------------------------------------------------
        # HSV
        # ----------------------------------------------------

        hsv_features = extract_hsv(image)

        # ----------------------------------------------------
        # LBP
        # ----------------------------------------------------

        lbp_features = extract_lbp(image)

        # ----------------------------------------------------
        # COMBINACIONES
        # ----------------------------------------------------

        hog_hsv = np.concatenate([
            hog_features,
            hsv_features
        ])

        full = np.concatenate([
            hog_features,
            hsv_features,
            lbp_features
        ])

        # ----------------------------------------------------
        # GUARDAR
        # ----------------------------------------------------

        X_rgb.append(rgb_features)

        X_hog.append(hog_features)

        X_hog_hsv.append(hog_hsv)

        X_full.append(full)

        y.append(label)

        filenames.append(path.name)

        if (i + 1) % 500 == 0:
            print(
                f"Procesadas: {i + 1}/{len(files)}"
            )

    return (
        np.array(X_rgb, dtype=np.float32),
        np.array(X_hog, dtype=np.float32),
        np.array(X_hog_hsv, dtype=np.float32),
        np.array(X_full, dtype=np.float32),
        np.array(y, dtype=np.int32),
        filenames
    )


# ============================================================
# EVALUAR MODELO
# ============================================================

def evaluate_model(
    name,
    model,
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\n")
    print("=" * 65)
    print(name)
    print("=" * 65)

    start_train = time.time()

    model.fit(
        X_train,
        y_train
    )

    train_time = time.time() - start_train

    start_predict = time.time()

    predictions = model.predict(
        X_test
    )

    inference_time = (
        time.time() - start_predict
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    inference_per_image = (
        inference_time / len(y_test)
    )

    print(
        f"Accuracy : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision: {precision * 100:.2f}%"
    )

    print(
        f"Recall   : {recall * 100:.2f}%"
    )

    print(
        f"F1       : {f1 * 100:.2f}%"
    )

    print(
        f"\nEntrenamiento: "
        f"{train_time:.3f} s"
    )

    print(
        f"Inferencia total: "
        f"{inference_time:.4f} s"
    )

    print(
        f"Inferencia/imagen: "
        f"{inference_per_image * 1000:.4f} ms"
    )

    print("\nMatriz de confusión:")

    print(cm)

    return {
        "Modelo": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Training_seconds": train_time,
        "Inference_ms_image":
            inference_per_image * 1000
    }


# ============================================================
# MAIN
# ============================================================

def main():

    (
        X_rgb,
        X_hog,
        X_hog_hsv,
        X_full,
        y,
        filenames
    ) = load_dataset()

    print("\n")
    print("=" * 65)
    print("DATASET")
    print("=" * 65)

    print(
        f"Total imágenes: {len(y)}"
    )

    print(
        f"No ship: {(y == 0).sum()}"
    )

    print(
        f"Ship: {(y == 1).sum()}"
    )

    # ========================================================
    # MISMO SPLIT PARA TODOS LOS MODELOS
    # ========================================================

    indices = np.arange(
        len(y)
    )

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    y_train = y[train_idx]
    y_test = y[test_idx]

    print(
        f"\nTrain: {len(train_idx)}"
    )

    print(
        f"Test : {len(test_idx)}"
    )

    results = []

    # ========================================================
    # MODELO 1
    # RGB + LOGISTIC REGRESSION
    # ========================================================

    model_rgb = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    results.append(
        evaluate_model(
            "RGB + Logistic Regression",
            model_rgb,
            X_rgb[train_idx],
            X_rgb[test_idx],
            y_train,
            y_test
        )
    )

    # ========================================================
    # MODELO 2
    # HOG + LINEAR SVM
    # ========================================================

    model_hog = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                random_state=42,
                max_iter=10000
            )
        )
    ])

    results.append(
        evaluate_model(
            "HOG + Linear SVM",
            model_hog,
            X_hog[train_idx],
            X_hog[test_idx],
            y_train,
            y_test
        )
    )

    # ========================================================
    # MODELO 3
    # HOG + HSV + SVM RBF
    # ========================================================

    model_hog_hsv = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                class_weight="balanced"
            )
        )
    ])

    results.append(
        evaluate_model(
            "HOG + HSV + SVM RBF",
            model_hog_hsv,
            X_hog_hsv[train_idx],
            X_hog_hsv[test_idx],
            y_train,
            y_test
        )
    )

    # ========================================================
    # MODELO 4
    # HOG + HSV + LBP + SVM RBF
    # ========================================================

    model_full = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                class_weight="balanced"
            )
        )
    ])

    results.append(
        evaluate_model(
            "HOG + HSV + LBP + SVM RBF",
            model_full,
            X_full[train_idx],
            X_full[test_idx],
            y_train,
            y_test
        )
    )

    # ========================================================
    # TABLA FINAL
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        by="Accuracy",
        ascending=False
    )

    print("\n")
    print("=" * 90)
    print("COMPARACIÓN FINAL")
    print("=" * 90)

    print(
        results_df[
            [
                "Modelo",
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "Inference_ms_image"
            ]
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # GUARDAR
    # ========================================================

    output = (
        RESULTS_DIR /
        "baseline_results.csv"
    )

    results_df.to_csv(
        output,
        index=False
    )

    print(
        f"\nResultados guardados en:\n{output}"
    )


if __name__ == "__main__":
    main()