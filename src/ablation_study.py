from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.model_selection import (
    StratifiedGroupKFold,
    GridSearchCV,
    cross_val_predict,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)


# ============================================================
# IMPORTS LOCALES
# ============================================================

try:
    from .features import (
        preprocess_image,
        extract_hog,
        extract_hsv,
    )

except ImportError:
    from features import (
        preprocess_image,
        extract_hog,
        extract_hsv,
    )


# ============================================================
# RUTAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = (
    BASE_DIR
    / "dataset"
    / "raw"
)

EXTERNAL_DIR = (
    BASE_DIR
    / "test_externo_balanceado_v3"
    / "images"
)

EXTERNAL_LABELS = (
    BASE_DIR
    / "test_externo_balanceado_v3"
    / "labels.csv"
)

RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "ablation"
)

CACHE_DIR = (
    RESULTS_DIR
    / "cache"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

RANDOM_STATE = 42

N_SPLITS = 5

FEATURE_MODES = [
    "HOG",
    "HSV",
    "HOG_HSV",
    "HOG_HSV_CLAHE",
]


# ============================================================
# SCENE ID
# ============================================================

def get_scene_id(filename):

    stem = Path(filename).stem

    parts = stem.split("__")

    if len(parts) >= 2:
        return parts[1]

    return "UNKNOWN"


# ============================================================
# CLAHE
# ============================================================

def apply_clahe_bgr(image):
    """
    Normalización local de iluminación.

    Se aplica CLAHE al canal L de LAB y luego
    se devuelve nuevamente a BGR.
    """

    image = preprocess_image(
        image
    )

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(
        lab
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l_norm = clahe.apply(
        l
    )

    lab_norm = cv2.merge([
        l_norm,
        a,
        b
    ])

    result = cv2.cvtColor(
        lab_norm,
        cv2.COLOR_LAB2BGR
    )

    return result


# ============================================================
# EXTRACCIÓN DE FEATURES
# ============================================================

def extract_mode_features(
    image,
    mode
):

    image = preprocess_image(
        image
    )

    if mode == "HOG":

        return extract_hog(
            image
        )

    if mode == "HSV":

        return extract_hsv(
            image
        )

    if mode == "HOG_HSV":

        hog_features = extract_hog(
            image
        )

        hsv_features = extract_hsv(
            image
        )

        return np.concatenate([
            hog_features,
            hsv_features
        ]).astype(
            np.float32
        )

    if mode == "HOG_HSV_CLAHE":

        normalized = apply_clahe_bgr(
            image
        )

        hog_features = extract_hog(
            normalized
        )

        hsv_features = extract_hsv(
            normalized
        )

        return np.concatenate([
            hog_features,
            hsv_features
        ]).astype(
            np.float32
        )

    raise ValueError(
        f"Modo no reconocido: {mode}"
    )


# ============================================================
# DATASET TRAIN
# ============================================================

def load_train_features(
    mode
):

    cache_file = (
        CACHE_DIR
        /
        f"train_{mode}.npz"
    )

    if cache_file.exists():

        print(
            f"  Cargando cache TRAIN: "
            f"{cache_file.name}"
        )

        data = np.load(
            cache_file,
            allow_pickle=True
        )

        return (
            data["X"],
            data["y"],
            data["groups"],
            data["filenames"]
        )

    files = sorted(
        TRAIN_DIR.glob(
            "*.png"
        )
    )

    if not files:

        raise RuntimeError(
            f"No hay imágenes en:\n"
            f"{TRAIN_DIR}"
        )

    X = []
    y = []
    groups = []
    filenames = []

    print(
        f"\nExtrayendo TRAIN [{mode}]..."
    )

    for i, path in enumerate(
        files
    ):

        label = int(
            path.name.split(
                "__"
            )[0]
        )

        image = cv2.imread(
            str(path)
        )

        if image is None:
            continue

        features = (
            extract_mode_features(
                image,
                mode
            )
        )

        X.append(
            features
        )

        y.append(
            label
        )

        groups.append(
            get_scene_id(
                path.name
            )
        )

        filenames.append(
            path.name
        )

        if (
            (i + 1) % 500
            == 0
        ):

            print(
                f"  {i + 1}"
                f"/"
                f"{len(files)}"
            )

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int32
    )

    groups = np.asarray(
        groups
    )

    filenames = np.asarray(
        filenames
    )

    np.savez_compressed(
        cache_file,
        X=X,
        y=y,
        groups=groups,
        filenames=filenames
    )

    return (
        X,
        y,
        groups,
        filenames
    )


# ============================================================
# DATASET EXTERNO
# ============================================================

def load_external_features(
    mode
):

    cache_file = (
        CACHE_DIR
        /
        f"external_{mode}.npz"
    )

    if cache_file.exists():

        print(
            f"  Cargando cache EXTERNAL: "
            f"{cache_file.name}"
        )

        data = np.load(
            cache_file,
            allow_pickle=True
        )

        return (
            data["X"],
            data["y"],
            data["filenames"]
        )

    if not EXTERNAL_LABELS.exists():

        raise FileNotFoundError(
            f"No existe:\n"
            f"{EXTERNAL_LABELS}"
        )

    labels_df = pd.read_csv(
        EXTERNAL_LABELS
    )

    required = {
        "filename",
        "label"
    }

    if not required.issubset(
        labels_df.columns
    ):

        raise ValueError(
            "labels.csv debe contener "
            "filename,label"
        )

    X = []
    y = []
    filenames = []

    print(
        f"\nExtrayendo EXTERNAL [{mode}]..."
    )

    for i, row in labels_df.iterrows():

        filename = str(
            row["filename"]
        )

        label = int(
            row["label"]
        )

        path = (
            EXTERNAL_DIR
            /
            filename
        )

        if not path.exists():
            continue

        image = cv2.imread(
            str(path)
        )

        if image is None:
            continue

        features = (
            extract_mode_features(
                image,
                mode
            )
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

        if (
            (i + 1) % 100
            == 0
        ):

            print(
                f"  {i + 1}"
                f"/"
                f"{len(labels_df)}"
            )

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int32
    )

    filenames = np.asarray(
        filenames
    )

    np.savez_compressed(
        cache_file,
        X=X,
        y=y,
        filenames=filenames
    )

    return (
        X,
        y,
        filenames
    )


# ============================================================
# MODELO
# ============================================================

def build_model():

    return Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            SVC(
                kernel="rbf",
                class_weight="balanced"
            )
        )
    ])


# ============================================================
# MÉTRICAS
# ============================================================

def calculate_metrics(
    y_true,
    scores,
    threshold
):

    predictions = (
        scores
        >
        threshold
    ).astype(
        np.int32
    )

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    try:

        auc = roc_auc_score(
            y_true,
            scores
        )

    except Exception:

        auc = np.nan

    return {

        "accuracy":
            accuracy_score(
                y_true,
                predictions
            ),

        "precision":
            precision_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                predictions
            ),

        "roc_auc":
            auc,

        "TN":
            int(
                cm[0, 0]
            ),

        "FP":
            int(
                cm[0, 1]
            ),

        "FN":
            int(
                cm[1, 0]
            ),

        "TP":
            int(
                cm[1, 1]
            ),
    }


# ============================================================
# BUSCAR UMBRAL
# ============================================================

def optimize_threshold(
    y_true,
    scores
):
    """
    El umbral se escoge EXCLUSIVAMENTE usando
    predicciones out-of-fold de ShipsNet.

    Criterio principal:
    Balanced Accuracy.

    En empate:
    F1.
    """

    minimum = min(
        -2.5,
        float(
            scores.min()
        )
    )

    maximum = max(
        2.5,
        float(
            scores.max()
        )
    )

    thresholds = np.linspace(
        minimum,
        maximum,
        1001
    )

    best_threshold = 0.0

    best_balanced = -1

    best_f1 = -1

    for threshold in thresholds:

        prediction = (
            scores
            >
            threshold
        ).astype(
            np.int32
        )

        balanced = (
            balanced_accuracy_score(
                y_true,
                prediction
            )
        )

        f1 = f1_score(
            y_true,
            prediction,
            zero_division=0
        )

        if (
            balanced
            >
            best_balanced
        ):

            best_balanced = (
                balanced
            )

            best_f1 = (
                f1
            )

            best_threshold = (
                threshold
            )

        elif (
            np.isclose(
                balanced,
                best_balanced
            )
            and
            f1 > best_f1
        ):

            best_f1 = f1

            best_threshold = (
                threshold
            )

    return float(
        best_threshold
    )


# ============================================================
# ANALIZAR UN MODELO
# ============================================================

def run_experiment(
    mode
):

    print("\n")
    print(
        "=" * 80
    )

    print(
        f"MODELO: {mode}"
    )

    print(
        "=" * 80
    )

    # ========================================================
    # FEATURES
    # ========================================================

    (
        X,
        y,
        groups,
        filenames
    ) = load_train_features(
        mode
    )

    (
        X_external,
        y_external,
        external_filenames
    ) = load_external_features(
        mode
    )

    print(
        f"\nFeatures: "
        f"{X.shape[1]}"
    )

    print(
        f"Train: "
        f"{len(y)}"
    )

    print(
        f"External: "
        f"{len(y_external)}"
    )


    # ========================================================
    # CV POR SCENE ID
    # ========================================================

    group_cv = (
        StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE
        )
    )


    # ========================================================
    # GRID SEARCH
    # ========================================================

    model = build_model()

    param_grid = {

        "classifier__C": [
            1,
            3,
            10
        ],

        "classifier__gamma": [
            "scale",
            0.001,
            0.01
        ]
    }

    scoring = {

        "accuracy":
            "accuracy",

        "f1":
            "f1",

        "balanced_accuracy":
            "balanced_accuracy",
    }

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        refit="balanced_accuracy",
        cv=group_cv,
        n_jobs=-1,
        verbose=0,
        return_train_score=False
    )

    print(
        "\nGridSearch..."
    )

    start = time.time()

    grid.fit(
        X,
        y,
        groups=groups
    )

    grid_time = (
        time.time()
        -
        start
    )

    print(
        f"Mejor C: "
        f"{grid.best_params_['classifier__C']}"
    )

    print(
        f"Mejor gamma: "
        f"{grid.best_params_['classifier__gamma']}"
    )

    print(
        "Mejor Balanced Acc CV: "
        f"{grid.best_score_ * 100:.3f}%"
    )


    # ========================================================
    # OOF DECISION SCORES
    # ========================================================

    best_model = (
        grid.best_estimator_
    )

    print(
        "\nGenerando scores OOF..."
    )

    oof_scores = cross_val_predict(
        best_model,
        X,
        y,
        cv=group_cv,
        groups=groups,
        n_jobs=-1,
        method="decision_function"
    )


    # ========================================================
    # OPTIMIZAR UMBRAL SOLO CON TRAIN OOF
    # ========================================================

    threshold = optimize_threshold(
        y,
        oof_scores
    )

    print(
        f"Umbral OOF: "
        f"{threshold:+.4f}"
    )

    oof_metrics = calculate_metrics(
        y,
        oof_scores,
        threshold
    )

    print(
        "\n--- OOF SHIPSNET ---"
    )

    print(
        f"Accuracy: "
        f"{oof_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{oof_metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Recall: "
        f"{oof_metrics['recall'] * 100:.2f}%"
    )

    print(
        f"F1: "
        f"{oof_metrics['f1'] * 100:.2f}%"
    )

    print(
        f"Balanced Acc: "
        f"{oof_metrics['balanced_accuracy'] * 100:.2f}%"
    )


    # ========================================================
    # ENTRENAR CON TODO SHIPSNET
    # ========================================================

    best_model.fit(
        X,
        y
    )


    # ========================================================
    # TEST EXTERNO
    # ========================================================

    external_scores = (
        best_model.decision_function(
            X_external
        )
    )

    external_metrics = (
        calculate_metrics(
            y_external,
            external_scores,
            threshold
        )
    )

    print(
        "\n--- TEST EXTERNO V3 ---"
    )

    print(
        f"Accuracy: "
        f"{external_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{external_metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Recall: "
        f"{external_metrics['recall'] * 100:.2f}%"
    )

    print(
        f"F1: "
        f"{external_metrics['f1'] * 100:.2f}%"
    )

    print(
        f"Balanced Acc: "
        f"{external_metrics['balanced_accuracy'] * 100:.2f}%"
    )

    print(
        f"ROC-AUC: "
        f"{external_metrics['roc_auc']:.3f}"
    )

    print(
        "\nMatriz:"
    )

    print([
        [
            external_metrics["TN"],
            external_metrics["FP"]
        ],
        [
            external_metrics["FN"],
            external_metrics["TP"]
        ]
    ])


    # ========================================================
    # GUARDAR PREDICCIONES EXTERNAS
    # ========================================================

    external_predictions = (
        external_scores
        >
        threshold
    ).astype(
        np.int32
    )

    external_df = pd.DataFrame({

        "filename":
            external_filenames,

        "real_label":
            y_external,

        "decision_score":
            external_scores,

        "threshold":
            threshold,

        "predicted_label":
            external_predictions
    })

    external_df.to_csv(
        RESULTS_DIR
        /
        f"predictions_external_{mode}.csv",
        index=False
    )


    # ========================================================
    # GUARDAR GRID
    # ========================================================

    grid_df = pd.DataFrame(
        grid.cv_results_
    )

    grid_columns = [
        "param_classifier__C",
        "param_classifier__gamma",
        "mean_test_accuracy",
        "mean_test_f1",
        "mean_test_balanced_accuracy",
        "std_test_balanced_accuracy",
        "mean_fit_time",
    ]

    grid_df[
        grid_columns
    ].sort_values(
        "mean_test_balanced_accuracy",
        ascending=False
    ).to_csv(
        RESULTS_DIR
        /
        f"grid_{mode}.csv",
        index=False
    )


    # ========================================================
    # RESUMEN
    # ========================================================

    result = {

        "model":
            mode,

        "n_features":
            X.shape[1],

        "best_C":
            grid.best_params_[
                "classifier__C"
            ],

        "best_gamma":
            grid.best_params_[
                "classifier__gamma"
            ],

        "threshold":
            threshold,

        "oof_accuracy":
            oof_metrics[
                "accuracy"
            ],

        "oof_precision":
            oof_metrics[
                "precision"
            ],

        "oof_recall":
            oof_metrics[
                "recall"
            ],

        "oof_f1":
            oof_metrics[
                "f1"
            ],

        "oof_balanced_accuracy":
            oof_metrics[
                "balanced_accuracy"
            ],

        "external_accuracy":
            external_metrics[
                "accuracy"
            ],

        "external_precision":
            external_metrics[
                "precision"
            ],

        "external_recall":
            external_metrics[
                "recall"
            ],

        "external_f1":
            external_metrics[
                "f1"
            ],

        "external_balanced_accuracy":
            external_metrics[
                "balanced_accuracy"
            ],

        "external_roc_auc":
            external_metrics[
                "roc_auc"
            ],

        "external_TN":
            external_metrics[
                "TN"
            ],

        "external_FP":
            external_metrics[
                "FP"
            ],

        "external_FN":
            external_metrics[
                "FN"
            ],

        "external_TP":
            external_metrics[
                "TP"
            ],

        "grid_seconds":
            grid_time,
    }

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "ABLATION STUDY - SHIP CLASSIFIER"
    )

    print(
        "=" * 80
    )

    results = []

    for mode in FEATURE_MODES:

        result = run_experiment(
            mode
        )

        results.append(
            result
        )


    # ========================================================
    # TABLA FINAL
    # ========================================================

    df = pd.DataFrame(
        results
    )

    df.to_csv(
        RESULTS_DIR
        /
        "ablation_summary.csv",
        index=False
    )


    # ========================================================
    # MOSTRAR RESUMEN
    # ========================================================

    print("\n")
    print(
        "=" * 110
    )

    print(
        "RESULTADO FINAL"
    )

    print(
        "=" * 110
    )

    display_columns = [

        "model",

        "oof_accuracy",
        "oof_f1",
        "oof_balanced_accuracy",

        "external_accuracy",
        "external_precision",
        "external_recall",
        "external_f1",
        "external_balanced_accuracy",
        "external_roc_auc",
    ]

    display_df = (
        df[
            display_columns
        ]
        .copy()
    )

    percentage_columns = [

        "oof_accuracy",
        "oof_f1",
        "oof_balanced_accuracy",

        "external_accuracy",
        "external_precision",
        "external_recall",
        "external_f1",
        "external_balanced_accuracy",
    ]

    for column in (
        percentage_columns
    ):

        display_df[
            column
        ] = (
            display_df[
                column
            ]
            *
            100
        )

    print(
        display_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.2f}"
        )
    )

    print(
        "\nResumen guardado en:"
    )

    print(
        RESULTS_DIR
        /
        "ablation_summary.csv"
    )


if __name__ == "__main__":

    main()