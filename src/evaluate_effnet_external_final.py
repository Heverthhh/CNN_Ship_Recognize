from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.models import efficientnet_b0

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
    / "cnn_efficientnet_b0_best.pt"
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
    / "efficientnet_external_final"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


IMG_SIZE = 128

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
# NORMALIZACIÓN
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


transform = transforms.Compose([

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
# MODELO
# ============================================================

def create_model():

    model = efficientnet_b0(
        weights=None
    )

    in_features = (
        model
        .classifier[1]
        .in_features
    )

    model.classifier[1] = (
        nn.Linear(
            in_features,
            2
        )
    )

    return model


# ============================================================
# TTA
# ============================================================

def generate_tta_images(image):

    return [

        image,

        TF.rotate(
            image,
            90
        ),

        TF.rotate(
            image,
            180
        ),

        TF.rotate(
            image,
            270
        ),

        TF.hflip(
            image
        ),

        TF.vflip(
            image
        ),
    ]


# ============================================================
# PREDICCIÓN NORMAL
# ============================================================

@torch.no_grad()
def predict_normal(
    model,
    image
):

    tensor = transform(
        image
    )

    tensor = (
        tensor
        .unsqueeze(0)
        .to(DEVICE)
    )

    logits = model(
        tensor
    )

    probability = torch.softmax(
        logits,
        dim=1
    )[0, 1]

    return float(
        probability.item()
    )


# ============================================================
# PREDICCIÓN TTA
# ============================================================

@torch.no_grad()
def predict_tta(
    model,
    image
):

    variants = generate_tta_images(
        image
    )

    tensors = [

        transform(
            variant
        )

        for variant
        in variants
    ]

    batch = torch.stack(
        tensors
    ).to(
        DEVICE
    )

    logits = model(
        batch
    )

    probabilities = torch.softmax(
        logits,
        dim=1
    )[:, 1]

    mean_probability = (
        probabilities
        .mean()
        .item()
    )

    std_probability = (
        probabilities
        .std()
        .item()
    )

    return (
        float(
            mean_probability
        ),
        float(
            std_probability
        )
    )


# ============================================================
# MÉTRICAS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities
):

    predictions = (
        probabilities
        >=
        THRESHOLD
    ).astype(
        np.int32
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

    auc = roc_auc_score(
        y_true,
        probabilities
    )

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[
            0,
            1
        ]
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

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
                tn
            ),

        "FP":
            int(
                fp
            ),

        "FN":
            int(
                fn
            ),

        "TP":
            int(
                tp
            ),

        "predictions":
            predictions,
    }


# ============================================================
# PRINT MÉTRICAS
# ============================================================

def print_metrics(
    title,
    metrics
):

    print("\n")
    print(
        "=" * 75
    )

    print(
        title
    )

    print(
        "=" * 75
    )

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
        f"{metrics['roc_auc']:.5f}"
    )

    print(
        "\nMatriz:"
    )

    print([

        [
            metrics[
                "TN"
            ],

            metrics[
                "FP"
            ]
        ],

        [
            metrics[
                "FN"
            ],

            metrics[
                "TP"
            ]
        ]
    ])


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "EFFICIENTNET-B0 - TEST EXTERNO FINAL"
    )

    print(
        "=" * 80
    )


    # ========================================================
    # DEVICE
    # ========================================================

    print(
        f"\nDevice: "
        f"{DEVICE}"
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
            f"No existe:\n"
            f"{MODEL_PATH}"
        )


    if not LABELS_PATH.exists():

        raise FileNotFoundError(
            f"No existe:\n"
            f"{LABELS_PATH}"
        )


    # ========================================================
    # CARGAR MODELO
    # ========================================================

    print(
        "\nCargando EfficientNet-B0..."
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
        f"\nCheckpoint:\n"
        f"{MODEL_PATH}"
    )

    print(
        f"\nThreshold congelado: "
        f"{THRESHOLD:.2f}"
    )


    # ========================================================
    # DATASET
    # ========================================================

    df = pd.read_csv(
        LABELS_PATH
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


    filenames = []

    y_true = []

    normal_probs = []

    tta_probs = []

    tta_stds = []


    normal_time = 0.0

    tta_time = 0.0


    # ========================================================
    # INFERENCIA
    # ========================================================

    print(
        "\nEjecutando inferencia normal + TTA..."
    )


    for index, row in df.iterrows():

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

        image = Image.open(
            path
        ).convert(
            "RGB"
        )


        # ====================================================
        # NORMAL
        # ====================================================

        if DEVICE.type == "cuda":

            torch.cuda.synchronize()


        start = time.perf_counter()


        normal_probability = (
            predict_normal(
                model,
                image
            )
        )


        if DEVICE.type == "cuda":

            torch.cuda.synchronize()


        normal_time += (

            time.perf_counter()

            -

            start
        )


        # ====================================================
        # TTA
        # ====================================================

        if DEVICE.type == "cuda":

            torch.cuda.synchronize()


        start = time.perf_counter()


        (
            tta_probability,
            tta_std
        ) = predict_tta(
            model,
            image
        )


        if DEVICE.type == "cuda":

            torch.cuda.synchronize()


        tta_time += (

            time.perf_counter()

            -

            start
        )


        # ====================================================
        # GUARDAR
        # ====================================================

        filenames.append(
            filename
        )

        y_true.append(
            label
        )

        normal_probs.append(
            normal_probability
        )

        tta_probs.append(
            tta_probability
        )

        tta_stds.append(
            tta_std
        )


        if (
            (index + 1)
            %
            50
            ==
            0
        ):

            print(
                f"{index + 1}"
                f"/"
                f"{len(df)}"
            )


    # ========================================================
    # ARRAYS
    # ========================================================

    y_true = np.asarray(
        y_true,
        dtype=np.int32
    )


    normal_probs = np.asarray(
        normal_probs,
        dtype=np.float64
    )


    tta_probs = np.asarray(
        tta_probs,
        dtype=np.float64
    )


    tta_stds = np.asarray(
        tta_stds,
        dtype=np.float64
    )


    # ========================================================
    # MÉTRICAS
    # ========================================================

    normal_metrics = (
        calculate_metrics(
            y_true,
            normal_probs
        )
    )


    tta_metrics = (
        calculate_metrics(
            y_true,
            tta_probs
        )
    )


    # ========================================================
    # RESULTADOS
    # ========================================================

    print_metrics(

        "EFFICIENTNET NORMAL",

        normal_metrics
    )


    print_metrics(

        "EFFICIENTNET + TTA",

        tta_metrics
    )


    # ========================================================
    # TIEMPOS
    # ========================================================

    normal_ms = (

        normal_time

        /

        len(y_true)

        *

        1000
    )


    tta_ms = (

        tta_time

        /

        len(y_true)

        *

        1000
    )


    print("\n")
    print(
        "=" * 75
    )

    print(
        "VELOCIDAD"
    )

    print(
        "=" * 75
    )


    print(
        f"Normal: "
        f"{normal_ms:.3f} ms/imagen"
    )


    print(
        f"TTA: "
        f"{tta_ms:.3f} ms/imagen"
    )


    # ========================================================
    # COMPARAR NORMAL VS TTA
    # ========================================================

    normal_correct = (

        normal_metrics[
            "predictions"
        ]

        ==

        y_true
    )


    tta_correct = (

        tta_metrics[
            "predictions"
        ]

        ==

        y_true
    )


    tta_fixed = (

        (~normal_correct)

        &

        tta_correct
    )


    tta_broke = (

        normal_correct

        &

        (~tta_correct)
    )


    print("\n")
    print(
        "=" * 75
    )

    print(
        "IMPACTO DE TTA"
    )

    print(
        "=" * 75
    )


    print(
        f"Errores normal: "
        f"{(~normal_correct).sum()}"
    )


    print(
        f"Errores TTA: "
        f"{(~tta_correct).sum()}"
    )


    print(
        f"TTA corrigió: "
        f"{tta_fixed.sum()}"
    )


    print(
        f"TTA dañó: "
        f"{tta_broke.sum()}"
    )


    # ========================================================
    # CSV COMPLETO
    # ========================================================

    results_df = pd.DataFrame({

        "filename":
            filenames,

        "real_label":
            y_true,

        "prob_effnet":
            normal_probs,

        "pred_effnet":
            normal_metrics[
                "predictions"
            ],

        "correct_effnet":
            normal_correct,

        "prob_effnet_tta":
            tta_probs,

        "tta_std":
            tta_stds,

        "pred_effnet_tta":
            tta_metrics[
                "predictions"
            ],

        "correct_effnet_tta":
            tta_correct,
    })


    results_df.to_csv(

        RESULTS_DIR
        /
        "efficientnet_final_predictions.csv",

        index=False
    )


    # ========================================================
    # ERRORES NORMAL
    # ========================================================

    normal_errors = (

        results_df[
            ~results_df[
                "correct_effnet"
            ]
        ]

        .copy()
    )


    normal_errors.to_csv(

        RESULTS_DIR
        /
        "efficientnet_final_errors.csv",

        index=False
    )


    # ========================================================
    # ERRORES TTA
    # ========================================================

    tta_errors = (

        results_df[
            ~results_df[
                "correct_effnet_tta"
            ]
        ]

        .copy()
    )


    tta_errors.to_csv(

        RESULTS_DIR
        /
        "efficientnet_tta_errors.csv",

        index=False
    )


    # ========================================================
    # CASOS CORREGIDOS POR TTA
    # ========================================================

    fixed_df = (

        results_df[
            tta_fixed
        ]

        .copy()
    )


    fixed_df.to_csv(

        RESULTS_DIR
        /
        "tta_fixed.csv",

        index=False
    )


    # ========================================================
    # CASOS DAÑADOS POR TTA
    # ========================================================

    broke_df = (

        results_df[
            tta_broke
        ]

        .copy()
    )


    broke_df.to_csv(

        RESULTS_DIR
        /
        "tta_broke.csv",

        index=False
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary_df = pd.DataFrame([

        {

            "model":
                "EfficientNet-B0",

            "mode":
                "normal",

            "accuracy":
                normal_metrics[
                    "accuracy"
                ],

            "precision":
                normal_metrics[
                    "precision"
                ],

            "recall":
                normal_metrics[
                    "recall"
                ],

            "f1":
                normal_metrics[
                    "f1"
                ],

            "balanced_accuracy":
                normal_metrics[
                    "balanced_accuracy"
                ],

            "roc_auc":
                normal_metrics[
                    "roc_auc"
                ],

            "TN":
                normal_metrics[
                    "TN"
                ],

            "FP":
                normal_metrics[
                    "FP"
                ],

            "FN":
                normal_metrics[
                    "FN"
                ],

            "TP":
                normal_metrics[
                    "TP"
                ],

            "ms_per_image":
                normal_ms,
        },

        {

            "model":
                "EfficientNet-B0",

            "mode":
                "TTA",

            "accuracy":
                tta_metrics[
                    "accuracy"
                ],

            "precision":
                tta_metrics[
                    "precision"
                ],

            "recall":
                tta_metrics[
                    "recall"
                ],

            "f1":
                tta_metrics[
                    "f1"
                ],

            "balanced_accuracy":
                tta_metrics[
                    "balanced_accuracy"
                ],

            "roc_auc":
                tta_metrics[
                    "roc_auc"
                ],

            "TN":
                tta_metrics[
                    "TN"
                ],

            "FP":
                tta_metrics[
                    "FP"
                ],

            "FN":
                tta_metrics[
                    "FN"
                ],

            "TP":
                tta_metrics[
                    "TP"
                ],

            "ms_per_image":
                tta_ms,
        }
    ])


    summary_df.to_csv(

        RESULTS_DIR
        /
        "efficientnet_final_summary.csv",

        index=False
    )


    # ========================================================
    # FINAL
    # ========================================================

    print("\n")
    print(
        "=" * 80
    )

    print(
        "COMPARACIÓN FINAL"
    )

    print(
        "=" * 80
    )


    print(
        f"HOG-SVM adaptado        : "
        f"89.00%"
    )


    print(
        f"MobileNetV3 normal      : "
        f"95.00%"
    )


    print(
        f"MobileNetV3 + TTA       : "
        f"96.00%"
    )


    print(
        f"EfficientNet-B0 normal  : "
        f"{normal_metrics['accuracy'] * 100:.2f}%"
    )


    print(
        f"EfficientNet-B0 + TTA   : "
        f"{tta_metrics['accuracy'] * 100:.2f}%"
    )


    print(
        f"\nResultados:\n"
        f"{RESULTS_DIR}"
    )


    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()