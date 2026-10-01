from pathlib import Path
import json
import time

import cv2
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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


# ------------------------------------------------------------
# SHIPSNET
# ------------------------------------------------------------

SHIPSNET_DIR = (
    BASE_DIR
    / "dataset"
    / "raw"
)


# ------------------------------------------------------------
# ADAPTACIÓN EXTERNA
# ------------------------------------------------------------

EXTERNAL_TRAIN_DIR = (
    BASE_DIR
    / "domain_adaptation"
    / "train"
    / "images"
)

EXTERNAL_TRAIN_LABELS = (
    BASE_DIR
    / "domain_adaptation"
    / "train"
    / "labels.csv"
)


EXTERNAL_VALID_DIR = (
    BASE_DIR
    / "domain_adaptation"
    / "valid"
    / "images"
)

EXTERNAL_VALID_LABELS = (
    BASE_DIR
    / "domain_adaptation"
    / "valid"
    / "labels.csv"
)


# ------------------------------------------------------------
# BENCHMARK V3
# ------------------------------------------------------------

V3_DIR = (
    BASE_DIR
    / "test_externo_balanceado_v3"
    / "images"
)

V3_LABELS = (
    BASE_DIR
    / "test_externo_balanceado_v3"
    / "labels.csv"
)


# ------------------------------------------------------------
# SALIDAS
# ------------------------------------------------------------

RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "domain_adaptation"
)

CACHE_DIR = (
    RESULTS_DIR
    / "cache"
)

MODELS_DIR = (
    BASE_DIR
    / "models"
)


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MODEL_PATH = (
    MODELS_DIR
    / "ship_classifier_hog_adapted.joblib"
)

CONFIG_PATH = (
    MODELS_DIR
    / "ship_classifier_hog_adapted_config.json"
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

RANDOM_STATE = 42


# ============================================================
# GRID
# ============================================================

C_VALUES = [
    3,
    10,
    30,
]

GAMMA_VALUES = [
    "scale",
    0.001,
]

# Peso adicional para las imágenes
# provenientes del dominio externo.
#
# ShipsNet siempre tiene peso = 1.0
#
# External train probará:
# 1.0
# 1.5
# 2.0

DOMAIN_WEIGHTS = [
    1.0,
    1.5,
    2.0,
]


# ============================================================
# EXTRAER HOG
# ============================================================

def extract_hog_from_path(
    path
):

    image = cv2.imread(
        str(path)
    )

    if image is None:

        raise ValueError(
            f"No se pudo abrir:\n{path}"
        )

    image = preprocess_image(
        image
    )

    features = extract_hog(
        image
    )

    return features.astype(
        np.float32
    )


# ============================================================
# CARGAR SHIPSNET
# ============================================================

def load_shipsnet():

    cache_file = (
        CACHE_DIR
        / "shipsnet_hog.npz"
    )

    if cache_file.exists():

        print(
            "Cargando HOG ShipsNet desde cache..."
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


    files = sorted(
        SHIPSNET_DIR.glob(
            "*.png"
        )
    )

    if not files:

        raise RuntimeError(
            f"No hay imágenes en:\n"
            f"{SHIPSNET_DIR}"
        )


    print("\n")
    print("=" * 70)
    print("EXTRAYENDO HOG - SHIPSNET")
    print("=" * 70)


    X = []
    y = []
    filenames = []


    for i, path in enumerate(
        files
    ):

        try:

            label = int(
                path.name
                .split("__")[0]
            )

        except Exception:

            continue


        if label not in [0, 1]:
            continue


        try:

            features = (
                extract_hog_from_path(
                    path
                )
            )

        except Exception as e:

            print(
                f"ERROR: {path.name}"
            )

            print(e)

            continue


        X.append(
            features
        )

        y.append(
            label
        )

        filenames.append(
            path.name
        )


        if (
            (i + 1)
            % 500
            ==
            0
        ):

            print(
                f"{i + 1}"
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
# CARGAR DATASET CSV
# ============================================================

def load_labeled_dataset(
    images_dir,
    labels_csv,
    cache_name
):

    cache_file = (
        CACHE_DIR
        /
        f"{cache_name}.npz"
    )


    if cache_file.exists():

        print(
            f"Cargando {cache_name} "
            f"desde cache..."
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


    if not labels_csv.exists():

        raise FileNotFoundError(
            f"No existe:\n"
            f"{labels_csv}"
        )


    df = pd.read_csv(
        labels_csv
    )


    if not {
        "filename",
        "label"
    }.issubset(
        df.columns
    ):

        raise ValueError(
            f"{labels_csv.name} debe contener "
            "filename,label"
        )


    print("\n")
    print("=" * 70)
    print(
        f"EXTRAYENDO HOG - "
        f"{cache_name.upper()}"
    )
    print("=" * 70)


    X = []
    y = []
    filenames = []


    for i, row in df.iterrows():

        filename = str(
            row["filename"]
        ).strip()

        try:

            label = int(
                row["label"]
            )

        except Exception:

            continue


        if label not in [0, 1]:
            continue


        path = (
            images_dir
            /
            filename
        )


        if not path.exists():

            print(
                f"No existe: "
                f"{filename}"
            )

            continue


        try:

            features = (
                extract_hog_from_path(
                    path
                )
            )

        except Exception as e:

            print(
                f"ERROR: {filename}"
            )

            print(e)

            continue


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
            (i + 1)
            % 200
            ==
            0
        ):

            print(
                f"{i + 1}"
                f"/"
                f"{len(df)}"
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
# CREAR MODELO
# ============================================================

def build_model(
    C,
    gamma
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


    accuracy = accuracy_score(
        y_true,
        predictions
    )


    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )


    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )


    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )


    balanced = (
        balanced_accuracy_score(
            y_true,
            predictions
        )
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
# OPTIMIZAR THRESHOLD
# ============================================================

def optimize_threshold(
    y_true,
    scores
):

    minimum = float(
        scores.min()
    )

    maximum = float(
        scores.max()
    )


    thresholds = np.linspace(
        minimum,
        maximum,
        2001
    )


    best_threshold = 0.0

    best_balanced = -1.0

    best_f1 = -1.0


    for threshold in thresholds:

        predictions = (
            scores
            >
            threshold
        ).astype(
            np.int32
        )


        balanced = (
            balanced_accuracy_score(
                y_true,
                predictions
            )
        )


        f1 = f1_score(
            y_true,
            predictions,
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

            best_f1 = (
                f1
            )

            best_threshold = (
                threshold
            )


    return float(
        best_threshold
    )


# ============================================================
# MOSTRAR MÉTRICAS
# ============================================================

def print_metrics(
    title,
    metrics
):

    print("\n")
    print("-" * 65)
    print(title)
    print("-" * 65)

    print(
        f"Accuracy          : "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision         : "
        f"{metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Recall            : "
        f"{metrics['recall'] * 100:.2f}%"
    )

    print(
        f"F1                : "
        f"{metrics['f1'] * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy : "
        f"{metrics['balanced_accuracy'] * 100:.2f}%"
    )

    print(
        f"ROC-AUC           : "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        "\nMatriz:"
    )

    print(
        [
            [
                metrics["TN"],
                metrics["FP"]
            ],
            [
                metrics["FN"],
                metrics["TP"]
            ]
        ]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "DOMAIN ADAPTATION - HOG + SVM"
    )

    print(
        "=" * 80
    )


    # ========================================================
    # 1. CARGAR DATASETS
    # ========================================================

    (
        X_shipsnet,
        y_shipsnet,
        shipsnet_files
    ) = load_shipsnet()


    (
        X_ext_train,
        y_ext_train,
        ext_train_files
    ) = load_labeled_dataset(

        EXTERNAL_TRAIN_DIR,

        EXTERNAL_TRAIN_LABELS,

        "external_train_hog"
    )


    (
        X_ext_valid,
        y_ext_valid,
        ext_valid_files
    ) = load_labeled_dataset(

        EXTERNAL_VALID_DIR,

        EXTERNAL_VALID_LABELS,

        "external_valid_hog"
    )


    (
        X_v3,
        y_v3,
        v3_files
    ) = load_labeled_dataset(

        V3_DIR,

        V3_LABELS,

        "external_v3_hog"
    )


    # ========================================================
    # RESUMEN
    # ========================================================

    print("\n")
    print("=" * 80)
    print("DATASETS")
    print("=" * 80)

    print(
        f"ShipsNet       : "
        f"{len(y_shipsnet)}"
    )

    print(
        f"External train : "
        f"{len(y_ext_train)}"
    )

    print(
        f"External valid : "
        f"{len(y_ext_valid)}"
    )

    print(
        f"External V3    : "
        f"{len(y_v3)}"
    )

    print(
        f"HOG features   : "
        f"{X_shipsnet.shape[1]}"
    )


    # ========================================================
    # 2. TRAIN COMBINADO
    # ========================================================

    X_train = np.concatenate([
        X_shipsnet,
        X_ext_train
    ])

    y_train = np.concatenate([
        y_shipsnet,
        y_ext_train
    ])


    # Dominio de cada muestra:
    #
    # 0 = ShipsNet
    # 1 = externo

    domain = np.concatenate([

        np.zeros(
            len(y_shipsnet),
            dtype=np.int32
        ),

        np.ones(
            len(y_ext_train),
            dtype=np.int32
        )

    ])


    print(
        f"\nTotal entrenamiento: "
        f"{len(y_train)}"
    )

    print(
        f"NO BARCO: "
        f"{(y_train == 0).sum()}"
    )

    print(
        f"BARCO: "
        f"{(y_train == 1).sum()}"
    )


    # ========================================================
    # 3. BÚSQUEDA DE CONFIGURACIÓN
    #
    # IMPORTANTE:
    # Solo se utiliza EXTERNAL VALID.
    #
    # V3 NO participa en la selección.
    # ========================================================

    experiments = []

    best_model = None

    best_threshold = None

    best_balanced = -1.0

    best_f1 = -1.0

    best_configuration = None


    total_experiments = (
        len(C_VALUES)
        *
        len(GAMMA_VALUES)
        *
        len(DOMAIN_WEIGHTS)
    )


    experiment_number = 0


    print("\n")
    print("=" * 80)
    print("OPTIMIZACIÓN SOBRE EXTERNAL VALID")
    print("=" * 80)

    print(
        f"Configuraciones: "
        f"{total_experiments}"
    )


    for C in C_VALUES:

        for gamma in GAMMA_VALUES:

            for domain_weight in (
                DOMAIN_WEIGHTS
            ):

                experiment_number += 1


                print(
                    f"\n[{experiment_number}"
                    f"/"
                    f"{total_experiments}] "
                    f"C={C} | "
                    f"gamma={gamma} | "
                    f"peso externo="
                    f"{domain_weight}"
                )


                # =================================================
                # PESOS
                # =================================================

                sample_weight = np.ones(
                    len(y_train),
                    dtype=np.float64
                )


                # Aumentar importancia del dominio externo
                sample_weight[
                    domain == 1
                ] = domain_weight


                # =================================================
                # MODELO
                # =================================================

                model = build_model(
                    C=C,
                    gamma=gamma
                )


                start = time.time()


                model.fit(

                    X_train,

                    y_train,

                    classifier__sample_weight=(
                        sample_weight
                    )
                )


                fit_seconds = (
                    time.time()
                    -
                    start
                )


                # =================================================
                # SCORES VALID
                # =================================================

                valid_scores = (
                    model.decision_function(
                        X_ext_valid
                    )
                )


                # =================================================
                # THRESHOLD SOLO VALID
                # =================================================

                threshold = (
                    optimize_threshold(
                        y_ext_valid,
                        valid_scores
                    )
                )


                valid_metrics = (
                    calculate_metrics(
                        y_ext_valid,
                        valid_scores,
                        threshold
                    )
                )


                print(
                    f"Threshold: "
                    f"{threshold:+.4f}"
                )

                print(
                    f"Balanced: "
                    f"{valid_metrics['balanced_accuracy'] * 100:.2f}%"
                )

                print(
                    f"F1: "
                    f"{valid_metrics['f1'] * 100:.2f}%"
                )


                # =================================================
                # GUARDAR EXPERIMENTO
                # =================================================

                experiments.append({

                    "C":
                        C,

                    "gamma":
                        gamma,

                    "domain_weight":
                        domain_weight,

                    "threshold":
                        threshold,

                    "valid_accuracy":
                        valid_metrics[
                            "accuracy"
                        ],

                    "valid_precision":
                        valid_metrics[
                            "precision"
                        ],

                    "valid_recall":
                        valid_metrics[
                            "recall"
                        ],

                    "valid_f1":
                        valid_metrics[
                            "f1"
                        ],

                    "valid_balanced_accuracy":
                        valid_metrics[
                            "balanced_accuracy"
                        ],

                    "valid_roc_auc":
                        valid_metrics[
                            "roc_auc"
                        ],

                    "fit_seconds":
                        fit_seconds,
                })


                # =================================================
                # MEJOR MODELO
                # =================================================

                current_balanced = (
                    valid_metrics[
                        "balanced_accuracy"
                    ]
                )

                current_f1 = (
                    valid_metrics[
                        "f1"
                    ]
                )


                if (
                    current_balanced
                    >
                    best_balanced
                ):

                    best_balanced = (
                        current_balanced
                    )

                    best_f1 = (
                        current_f1
                    )

                    best_model = (
                        model
                    )

                    best_threshold = (
                        threshold
                    )

                    best_configuration = {

                        "C":
                            C,

                        "gamma":
                            gamma,

                        "domain_weight":
                            domain_weight
                    }


                elif (
                    np.isclose(
                        current_balanced,
                        best_balanced
                    )
                    and
                    current_f1
                    >
                    best_f1
                ):

                    best_f1 = (
                        current_f1
                    )

                    best_model = (
                        model
                    )

                    best_threshold = (
                        threshold
                    )

                    best_configuration = {

                        "C":
                            C,

                        "gamma":
                            gamma,

                        "domain_weight":
                            domain_weight
                    }


    # ========================================================
    # 4. GUARDAR GRID
    # ========================================================

    experiments_df = (
        pd.DataFrame(
            experiments
        )
    )


    experiments_df = (
        experiments_df
        .sort_values(
            [
                "valid_balanced_accuracy",
                "valid_f1"
            ],
            ascending=False
        )
    )


    experiments_df.to_csv(

        RESULTS_DIR
        /
        "domain_adaptation_grid.csv",

        index=False
    )


    # ========================================================
    # 5. MEJOR CONFIGURACIÓN
    # ========================================================

    print("\n")
    print("=" * 80)
    print("MEJOR CONFIGURACIÓN")
    print("=" * 80)

    print(
        f"C = "
        f"{best_configuration['C']}"
    )

    print(
        f"gamma = "
        f"{best_configuration['gamma']}"
    )

    print(
        f"peso dominio externo = "
        f"{best_configuration['domain_weight']}"
    )

    print(
        f"threshold = "
        f"{best_threshold:+.4f}"
    )


    # ========================================================
    # 6. VALIDACIÓN
    # ========================================================

    valid_scores = (
        best_model
        .decision_function(
            X_ext_valid
        )
    )


    valid_metrics = (
        calculate_metrics(
            y_ext_valid,
            valid_scores,
            best_threshold
        )
    )


    print_metrics(
        "EXTERNAL VALID",
        valid_metrics
    )


    # ========================================================
    # 7. EVALUACIÓN V3
    #
    # V3 se evalúa SOLO después de seleccionar
    # completamente modelo y threshold.
    # ========================================================

    print("\n")
    print("=" * 80)
    print("EVALUANDO BENCHMARK V3")
    print("=" * 80)


    v3_scores = (
        best_model
        .decision_function(
            X_v3
        )
    )


    v3_metrics = (
        calculate_metrics(
            y_v3,
            v3_scores,
            best_threshold
        )
    )


    print_metrics(
        "EXTERNAL V3",
        v3_metrics
    )


    # ========================================================
    # 8. PREDICCIONES V3
    # ========================================================

    v3_predictions = (
        v3_scores
        >
        best_threshold
    ).astype(
        np.int32
    )


    predictions_df = pd.DataFrame({

        "filename":
            v3_files,

        "real_label":
            y_v3,

        "decision_score":
            v3_scores,

        "threshold":
            best_threshold,

        "predicted_label":
            v3_predictions,

        "correct":
            (
                y_v3
                ==
                v3_predictions
            )
    })


    predictions_df.to_csv(

        RESULTS_DIR
        /
        "domain_adapted_v3_predictions.csv",

        index=False
    )


    # ========================================================
    # 9. GUARDAR MODELO
    # ========================================================

    joblib.dump(
        best_model,
        MODEL_PATH
    )


    # ========================================================
    # 10. CONFIG JSON
    # ========================================================

    config = {

        "descriptor":
            "HOG",

        "classifier":
            "SVM RBF",

        "C":
            best_configuration[
                "C"
            ],

        "gamma":
            best_configuration[
                "gamma"
            ],

        "external_domain_weight":
            best_configuration[
                "domain_weight"
            ],

        "threshold":
            float(
                best_threshold
            ),

        "training_samples":
            int(
                len(y_train)
            ),

        "shipsnet_samples":
            int(
                len(y_shipsnet)
            ),

        "external_train_samples":
            int(
                len(y_ext_train)
            ),

        "validation_samples":
            int(
                len(y_ext_valid)
            ),

        "validation_metrics": {

            key:
                float(value)

            for key, value
            in valid_metrics.items()
        },

        "v3_metrics": {

            key:
                float(value)

            for key, value
            in v3_metrics.items()
        }
    }


    with open(
        CONFIG_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            config,
            file,
            indent=4
        )


    # ========================================================
    # 11. RESUMEN CSV
    # ========================================================

    summary = pd.DataFrame([{

        "descriptor":
            "HOG",

        "classifier":
            "SVM RBF",

        "C":
            best_configuration[
                "C"
            ],

        "gamma":
            best_configuration[
                "gamma"
            ],

        "external_domain_weight":
            best_configuration[
                "domain_weight"
            ],

        "threshold":
            best_threshold,

        "valid_accuracy":
            valid_metrics[
                "accuracy"
            ],

        "valid_precision":
            valid_metrics[
                "precision"
            ],

        "valid_recall":
            valid_metrics[
                "recall"
            ],

        "valid_f1":
            valid_metrics[
                "f1"
            ],

        "valid_balanced_accuracy":
            valid_metrics[
                "balanced_accuracy"
            ],

        "valid_roc_auc":
            valid_metrics[
                "roc_auc"
            ],

        "v3_accuracy":
            v3_metrics[
                "accuracy"
            ],

        "v3_precision":
            v3_metrics[
                "precision"
            ],

        "v3_recall":
            v3_metrics[
                "recall"
            ],

        "v3_f1":
            v3_metrics[
                "f1"
            ],

        "v3_balanced_accuracy":
            v3_metrics[
                "balanced_accuracy"
            ],

        "v3_roc_auc":
            v3_metrics[
                "roc_auc"
            ],

        "v3_TN":
            v3_metrics[
                "TN"
            ],

        "v3_FP":
            v3_metrics[
                "FP"
            ],

        "v3_FN":
            v3_metrics[
                "FN"
            ],

        "v3_TP":
            v3_metrics[
                "TP"
            ],
    }])


    summary.to_csv(

        RESULTS_DIR
        /
        "domain_adapted_summary.csv",

        index=False
    )


    # ========================================================
    # FINAL
    # ========================================================

    print("\n")
    print("=" * 80)
    print("MODELO GUARDADO")
    print("=" * 80)

    print(
        f"\n{MODEL_PATH}"
    )

    print(
        f"\nConfiguración:\n"
        f"{CONFIG_PATH}"
    )

    print(
        f"\nResumen:\n"
        f"{RESULTS_DIR / 'domain_adapted_summary.csv'}"
    )

    print(
        "\nIMPORTANTE:"
    )

    print(
        "V3 no fue utilizado para entrenar ni para "
        "seleccionar hiperparámetros/threshold."
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()