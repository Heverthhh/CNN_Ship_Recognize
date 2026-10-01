from pathlib import Path
import time
import json

import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import mobilenet_v3_small

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
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "cnn_mobilenetv3_best.pt"
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
    / "cnn_external_final"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMG_SIZE = 96

BATCH_SIZE = 64

NUM_WORKERS = 4

THRESHOLD = 0.5


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# TRANSFORMACIÓN
# EXACTAMENTE LA MISMA DE VALIDACIÓN
# ============================================================

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


test_transform = transforms.Compose([

    transforms.Resize(
        (
            IMG_SIZE,
            IMG_SIZE
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ),
])


# ============================================================
# DATASET
# ============================================================

class ExternalTestDataset(Dataset):

    def __init__(
        self,
        dataframe,
        images_dir,
        transform
    ):

        self.df = (
            dataframe
            .reset_index(
                drop=True
            )
        )

        self.images_dir = (
            images_dir
        )

        self.transform = (
            transform
        )


    def __len__(self):

        return len(
            self.df
        )


    def __getitem__(
        self,
        index
    ):

        row = self.df.iloc[
            index
        ]

        filename = str(
            row["filename"]
        )

        label = int(
            row["label"]
        )

        path = (
            self.images_dir
            /
            filename
        )

        image = Image.open(
            path
        ).convert(
            "RGB"
        )

        image = self.transform(
            image
        )

        return (
            image,
            label,
            filename
        )


# ============================================================
# MODELO
# ============================================================

def create_model():

    # No descargamos pesos ImageNet.
    # Vamos a cargar directamente nuestro checkpoint.
    model = mobilenet_v3_small(
        weights=None
    )

    in_features = (
        model.classifier[3]
        .in_features
    )

    model.classifier[3] = (
        nn.Linear(
            in_features,
            2
        )
    )

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 78
    )

    print(
        "EVALUACIÓN CNN - TEST EXTERNO FINAL"
    )

    print(
        "=" * 78
    )


    # ========================================================
    # DEVICE
    # ========================================================

    print(
        f"\nDevice: {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )


    # ========================================================
    # VERIFICACIONES
    # ========================================================

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"No existe el modelo:\n"
            f"{MODEL_PATH}"
        )


    if not LABELS_PATH.exists():

        raise FileNotFoundError(
            f"No existe:\n"
            f"{LABELS_PATH}"
        )


    # ========================================================
    # CARGAR CSV
    # ========================================================

    df = pd.read_csv(
        LABELS_PATH
    )


    required_columns = {
        "filename",
        "label"
    }


    if not required_columns.issubset(
        df.columns
    ):

        raise ValueError(
            "labels.csv debe contener "
            "filename y label"
        )


    print(
        f"\nImágenes: "
        f"{len(df)}"
    )

    print(
        f"BARCO: "
        f"{(df['label'] == 1).sum()}"
    )

    print(
        f"NO BARCO: "
        f"{(df['label'] == 0).sum()}"
    )


    # ========================================================
    # DATALOADER
    # ========================================================

    dataset = ExternalTestDataset(

        df,

        TEST_DIR,

        test_transform
    )


    loader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=(
            DEVICE.type
            ==
            "cuda"
        ),

        persistent_workers=(
            NUM_WORKERS > 0
        )
    )


    # ========================================================
    # CARGAR MODELO
    # ========================================================

    print(
        "\nCargando MobileNetV3-Small..."
    )


    model = create_model()

    state_dict = torch.load(

        MODEL_PATH,

        map_location=DEVICE,

        weights_only=True
    )


    model.load_state_dict(
        state_dict
    )


    model = model.to(
        DEVICE
    )

    model.eval()


    print(
        f"Modelo:\n{MODEL_PATH}"
    )

    print(
        f"\nThreshold congelado: "
        f"{THRESHOLD:.2f}"
    )


    # ========================================================
    # INFERENCIA
    # ========================================================

    all_true = []

    all_pred = []

    all_prob_ship = []

    all_filenames = []

    total_inference_time = 0.0


    print(
        "\nEjecutando inferencia..."
    )


    with torch.no_grad():

        for images, labels, filenames in loader:

            images = images.to(

                DEVICE,

                non_blocking=True
            )


            # ----------------------------------------------
            # SINCRONIZAR GPU PARA MEDIR TIEMPO REAL
            # ----------------------------------------------

            if DEVICE.type == "cuda":

                torch.cuda.synchronize()


            start = time.perf_counter()


            logits = model(
                images
            )


            if DEVICE.type == "cuda":

                torch.cuda.synchronize()


            elapsed = (
                time.perf_counter()
                -
                start
            )


            total_inference_time += (
                elapsed
            )


            probabilities = torch.softmax(

                logits,

                dim=1

            )[:, 1]


            predictions = (

                probabilities

                >= THRESHOLD

            ).long()


            all_true.extend(
                labels.numpy()
            )


            all_pred.extend(

                predictions
                .cpu()
                .numpy()
            )


            all_prob_ship.extend(

                probabilities
                .cpu()
                .numpy()
            )


            all_filenames.extend(
                filenames
            )


    # ========================================================
    # ARRAYS
    # ========================================================

    y_true = np.asarray(
        all_true,
        dtype=np.int32
    )


    y_pred = np.asarray(
        all_pred,
        dtype=np.int32
    )


    probabilities = np.asarray(
        all_prob_ship,
        dtype=np.float64
    )


    # ========================================================
    # MÉTRICAS
    # ========================================================

    accuracy = accuracy_score(
        y_true,
        y_pred
    )


    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )


    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )


    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )


    balanced = (
        balanced_accuracy_score(
            y_true,
            y_pred
        )
    )


    auc = roc_auc_score(
        y_true,
        probabilities
    )


    cm = confusion_matrix(

        y_true,

        y_pred,

        labels=[
            0,
            1
        ]
    )


    tn, fp, fn, tp = (
        cm.ravel()
    )


    # ========================================================
    # TIEMPO
    # ========================================================

    ms_per_image = (

        total_inference_time
        /
        len(y_true)

        *

        1000.0
    )


    images_per_second = (

        len(y_true)

        /
        total_inference_time

        if total_inference_time > 0

        else 0
    )


    # ========================================================
    # RESULTADOS
    # ========================================================

    print("\n")
    print(
        "=" * 78
    )

    print(
        "RESULTADO CNN - TEST EXTERNO FINAL"
    )

    print(
        "=" * 78
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


    print([
        [
            int(tn),
            int(fp)
        ],
        [
            int(fn),
            int(tp)
        ]
    ])


    print(
        f"\nTiempo CNN/img    : "
        f"{ms_per_image:.3f} ms"
    )


    print(
        f"Rendimiento GPU   : "
        f"{images_per_second:.1f} img/s"
    )


    # ========================================================
    # ESTADÍSTICAS DE PROBABILIDAD
    # ========================================================

    probs_no_ship = (
        probabilities[
            y_true == 0
        ]
    )


    probs_ship = (
        probabilities[
            y_true == 1
        ]
    )


    print(
        "\nProbabilidad P(BARCO)"
    )


    print(
        f"NO BARCO - mediana: "
        f"{np.median(probs_no_ship):.4f}"
    )


    print(
        f"BARCO    - mediana: "
        f"{np.median(probs_ship):.4f}"
    )


    # ========================================================
    # PREDICCIONES CSV
    # ========================================================

    results_df = pd.DataFrame({

        "filename":
            all_filenames,

        "real_label":
            y_true,

        "prob_ship":
            probabilities,

        "threshold":
            THRESHOLD,

        "predicted_label":
            y_pred,

        "correct":
            (
                y_true
                ==
                y_pred
            ),
    })


    results_df.to_csv(

        RESULTS_DIR
        /
        "cnn_final_predictions.csv",

        index=False
    )


    # ========================================================
    # ERRORES
    # ========================================================

    errors_df = (

        results_df[
            results_df[
                "correct"
            ] == False
        ]

        .copy()
    )


    errors_df.to_csv(

        RESULTS_DIR
        /
        "cnn_final_errors.csv",

        index=False
    )


    # ========================================================
    # RESUMEN
    # ========================================================

    summary_df = pd.DataFrame([{

        "model":
            "MobileNetV3-Small",

        "threshold":
            THRESHOLD,

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

        "ms_per_image":
            ms_per_image,

        "images_per_second":
            images_per_second,
    }])


    summary_df.to_csv(

        RESULTS_DIR
        /
        "cnn_final_summary.csv",

        index=False
    )


    # ========================================================
    # JSON
    # ========================================================

    summary_json = {

        "model":
            "MobileNetV3-Small",

        "input_size":
            IMG_SIZE,

        "threshold":
            THRESHOLD,

        "metrics": {

            "accuracy":
                float(
                    accuracy
                ),

            "precision":
                float(
                    precision
                ),

            "recall":
                float(
                    recall
                ),

            "f1":
                float(
                    f1
                ),

            "balanced_accuracy":
                float(
                    balanced
                ),

            "roc_auc":
                float(
                    auc
                ),
        },

        "confusion_matrix": [

            [
                int(tn),
                int(fp)
            ],

            [
                int(fn),
                int(tp)
            ]
        ],

        "performance": {

            "ms_per_image":
                float(
                    ms_per_image
                ),

            "images_per_second":
                float(
                    images_per_second
                ),
        },
    }


    with open(

        RESULTS_DIR
        /
        "cnn_final_summary.json",

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            summary_json,

            file,

            indent=4
        )


    # ========================================================
    # FINAL
    # ========================================================

    print(
        f"\nErrores totales: "
        f"{len(errors_df)}"
        f"/"
        f"{len(results_df)}"
    )


    print(
        f"\nResultados guardados en:\n"
        f"{RESULTS_DIR}"
    )


    print(
        "=" * 78
    )


if __name__ == "__main__":

    main()