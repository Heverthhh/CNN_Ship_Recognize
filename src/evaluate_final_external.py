from pathlib import Path
import json

import cv2
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

try:

    from .features import (
        preprocess_image,
        extract_hog,
    )

except ImportError:

    from features import (
        preprocess_image,
        extract_hog,
    )


# ============================================================
# RUTAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent


MODEL_PATH = (
    BASE_DIR
    / "models"
    / "ship_classifier_hog_adapted.joblib"
)


CONFIG_PATH = (
    BASE_DIR
    / "models"
    / "ship_classifier_hog_adapted_config.json"
)


TEST_DIR = (
    BASE_DIR
    / "test_externo_final"
    / "images"
)


LABELS_PATH = (
    BASE_DIR
    / "test_externo_final"
    / "labels.csv"
)


RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "final_external"
)


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HOG
# ============================================================

def extract_features(
    path
):

    image = cv2.imread(
        str(path)
    )

    if image is None:

        raise ValueError(
            f"No se pudo abrir {path}"
        )

    image = preprocess_image(
        image
    )

    return extract_hog(
        image
    ).astype(
        np.float32
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 75
    )

    print(
        "EVALUACIÓN FINAL - HOG SVM ADAPTADO"
    )

    print(
        "=" * 75
    )


    # ========================================================
    # MODELO
    # ========================================================

    model = joblib.load(
        MODEL_PATH
    )


    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        config = json.load(
            file
        )


    threshold = float(
        config[
            "threshold"
        ]
    )


    print(
        f"\nThreshold congelado: "
        f"{threshold:+.6f}"
    )


    print(
        f"C congelado: "
        f"{config['C']}"
    )


    print(
        f"Gamma congelado: "
        f"{config['gamma']}"
    )


    # ========================================================
    # DATASET
    # ========================================================

    df = pd.read_csv(
        LABELS_PATH
    )


    X = []
    y = []
    filenames = []


    print(
        f"\nProcesando "
        f"{len(df)} imágenes..."
    )


    for i, row in df.iterrows():

        filename = str(
            row[
                "filename"
            ]
        )


        label = int(
            row[
                "label"
            ]
        )


        path = (
            TEST_DIR
            /
            filename
        )


        features = extract_features(
            path
        )


        X.append(
            features
        )

        y.append(
            label
        )

        filenames.append(
            filename
        )


    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int32
    )


    # ========================================================
    # INFERENCIA
    # ========================================================

    scores = model.decision_function(
        X
    )


    predictions = (
        scores
        >
        threshold
    ).astype(
        np.int32
    )


    # ========================================================
    # MÉTRICAS
    # ========================================================

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


    balanced = balanced_accuracy_score(
        y,
        predictions
    )


    auc = roc_auc_score(
        y,
        scores
    )


    cm = confusion_matrix(
        y,
        predictions,
        labels=[0, 1]
    )


    tn, fp, fn, tp = (
        cm.ravel()
    )


    print("\n")
    print(
        "=" * 75
    )

    print(
        "RESULTADO TEST EXTERNO FINAL"
    )

    print(
        "=" * 75
    )


    print(
        f"Accuracy          : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Precision         : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall            : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"F1                : "
        f"{f1 * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy : "
        f"{balanced * 100:.2f}%"
    )

    print(
        f"ROC-AUC           : "
        f"{auc:.4f}"
    )


    print(
        "\nMatriz de confusión:"
    )

    print(
        [
            [
                int(tn),
                int(fp)
            ],
            [
                int(fn),
                int(tp)
            ]
        ]
    )


    # ========================================================
    # GUARDAR PREDICCIONES
    # ========================================================

    results_df = pd.DataFrame({

        "filename":
            filenames,

        "real_label":
            y,

        "decision_score":
            scores,

        "threshold":
            threshold,

        "predicted_label":
            predictions,

        "correct":
            (
                y
                ==
                predictions
            ),
    })


    results_df.to_csv(

        RESULTS_DIR
        /
        "final_predictions.csv",

        index=False
    )


    summary_df = pd.DataFrame([{

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "balanced_accuracy":
            balanced,

        "roc_auc":
            auc,

        "TN":
            int(tn),

        "FP":
            int(fp),

        "FN":
            int(fn),

        "TP":
            int(tp),

        "threshold":
            threshold,
    }])


    summary_df.to_csv(

        RESULTS_DIR
        /
        "final_summary.csv",

        index=False
    )


    print(
        f"\nGuardado en:\n"
        f"{RESULTS_DIR}"
    )


    print(
        "=" * 75
    )


if __name__ == "__main__":

    main()