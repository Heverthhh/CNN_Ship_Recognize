from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import functional as TF
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
    / "cnn_tta"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMG_SIZE = 96

THRESHOLD = 0.5


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


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
# VARIANTES TTA
# ============================================================

def generate_tta_images(image):

    return [

        # Original
        image,

        # Rotaciones
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

        # Flips
        TF.hflip(
            image
        ),

        TF.vflip(
            image
        ),
    ]


# ============================================================
# INFERENCIA TTA
# ============================================================

@torch.no_grad()
def predict_tta(
    model,
    image
):

    variants = (
        generate_tta_images(
            image
        )
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


    # También guardamos dispersión:
    # útil para detectar incertidumbre.
    std_probability = (
        probabilities
        .std()
        .item()
    )


    return (
        mean_probability,
        std_probability
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 75
    )

    print(
        "MOBILENETV3 - TEST TIME AUGMENTATION"
    )

    print(
        "=" * 75
    )


    print(
        f"\nDevice: {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )


    # ========================================================
    # MODELO
    # ========================================================

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


    # ========================================================
    # DATASET
    # ========================================================

    df = pd.read_csv(
        LABELS_PATH
    )


    filenames = []

    y_true = []

    probabilities = []

    std_probabilities = []


    print(
        f"\nProcesando "
        f"{len(df)} imágenes..."
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


        (
            probability,
            std_probability
        ) = predict_tta(
            model,
            image
        )


        filenames.append(
            filename
        )

        y_true.append(
            label
        )

        probabilities.append(
            probability
        )

        std_probabilities.append(
            std_probability
        )


        if (
            (index + 1)
            % 50
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


    probabilities = np.asarray(
        probabilities,
        dtype=np.float64
    )


    predictions = (
        probabilities
        >=
        THRESHOLD
    ).astype(
        np.int32
    )


    # ========================================================
    # MÉTRICAS
    # ========================================================

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
        labels=[0, 1]
    )


    tn, fp, fn, tp = (
        cm.ravel()
    )


    # ========================================================
    # RESULTADOS
    # ========================================================

    print("\n")
    print(
        "=" * 75
    )

    print(
        "RESULTADO TTA"
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
        f"{auc:.5f}"
    )


    print(
        "\nMatriz:"
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


    # ========================================================
    # CSV
    # ========================================================

    results = pd.DataFrame({

        "filename":
            filenames,

        "real_label":
            y_true,

        "prob_ship_tta":
            probabilities,

        "tta_std":
            std_probabilities,

        "threshold":
            THRESHOLD,

        "predicted_label":
            predictions,

        "correct":
            (
                y_true
                ==
                predictions
            ),
    })


    results.to_csv(

        RESULTS_DIR
        /
        "tta_predictions.csv",

        index=False
    )


    errors = results[
        results[
            "correct"
        ] == False
    ]


    errors.to_csv(

        RESULTS_DIR
        /
        "tta_errors.csv",

        index=False
    )


    print(
        f"\nErrores: "
        f"{len(errors)}"
        f"/"
        f"{len(results)}"
    )


    print(
        f"\nResultados:"
        f"\n{RESULTS_DIR}"
    )


    print(
        "=" * 75
    )


if __name__ == "__main__":

    main()