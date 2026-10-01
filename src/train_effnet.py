from pathlib import Path
import random
import time
import json

import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
)

from sklearn.model_selection import StratifiedGroupKFold
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
# CONFIGURACIÓN GENERAL
# ============================================================

SEED = 42

BASE_DIR = Path(__file__).resolve().parent.parent

SHIPSNET_DIR = (
    BASE_DIR
    / "dataset"
    / "raw"
)


EXTERNAL_TRAIN_DIR = (
    BASE_DIR
    / "domain_adaptation"
    / "train"
    / "images"
)

EXTERNAL_TRAIN_CSV = (
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

EXTERNAL_VALID_CSV = (
    BASE_DIR
    / "domain_adaptation"
    / "valid"
    / "labels.csv"
)


RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "efficientnet"
)

MODELS_DIR = (
    BASE_DIR
    / "models"
)


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


BEST_MODEL_PATH = (
    MODELS_DIR
    / "cnn_efficientnet_b0_best.pt"
)

BEST_SHIPSNET_MODEL_PATH = (
    MODELS_DIR
    / "cnn_efficientnet_b0_best_shipsnet.pt"
)


# ============================================================
# HIPERPARÁMETROS
# ============================================================

# Aunque las imágenes originales son 80x80,
# EfficientNet trabaja mejor con algo más de resolución.
IMG_SIZE = 128

# RTX 4060 8 GB
BATCH_SIZE = 32

HEAD_EPOCHS = 5

FINETUNE_EPOCHS = 40

HEAD_LR = 1e-3

FINETUNE_LR = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 4

PATIENCE = 10

LABEL_SMOOTHING = 0.05


# ============================================================
# PESO DE LOS DOS DOMINIOS PARA SELECCIONAR CHECKPOINT
# ============================================================

SHIPSNET_SELECTION_WEIGHT = 0.75

EXTERNAL_SELECTION_WEIGHT = 0.25


# ============================================================
# REPRODUCIBILIDAD
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False

    torch.backends.cudnn.benchmark = True


set_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# NORMALIZACIÓN IMAGENET
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


# ============================================================
# DATA AUGMENTATION
# ============================================================

train_transform = transforms.Compose([

    # --------------------------------------------------------
    # Crop y zoom aleatorio
    # --------------------------------------------------------

    transforms.RandomResizedCrop(
        IMG_SIZE,
        scale=(0.75, 1.0),
        ratio=(0.88, 1.12)
    ),

    # --------------------------------------------------------
    # En imágenes aéreas la orientación no importa
    # --------------------------------------------------------

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomVerticalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=180
    ),

    # --------------------------------------------------------
    # Robustez frente a distintos sensores / iluminación
    # --------------------------------------------------------

    transforms.ColorJitter(
        brightness=0.25,
        contrast=0.25,
        saturation=0.25,
        hue=0.05
    ),

    # --------------------------------------------------------
    # Blur ocasional
    # --------------------------------------------------------

    transforms.RandomApply(

        [
            transforms.GaussianBlur(
                kernel_size=3,
                sigma=(0.1, 1.0)
            )
        ],

        p=0.15
    ),

    # --------------------------------------------------------
    # Tensor
    # --------------------------------------------------------

    transforms.ToTensor(),

    # --------------------------------------------------------
    # Normalización ImageNet
    # --------------------------------------------------------

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ),
])


# ============================================================
# VALIDACIÓN SIN AUGMENTATION
# ============================================================

valid_transform = transforms.Compose([

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

class ShipDataset(Dataset):

    def __init__(
        self,
        records,
        transform
    ):

        self.records = records

        self.transform = transform


    def __len__(self):

        return len(
            self.records
        )


    def __getitem__(
        self,
        index
    ):

        record = self.records[
            index
        ]

        path = record[
            "path"
        ]

        label = int(
            record[
                "label"
            ]
        )


        image = Image.open(
            path
        ).convert(
            "RGB"
        )


        if self.transform is not None:

            image = self.transform(
                image
            )


        return (
            image,
            label
        )


# ============================================================
# CARGAR SHIPSNET
# ============================================================

def load_shipsnet_records():

    records = []

    files = sorted(
        SHIPSNET_DIR.glob(
            "*.png"
        )
    )


    if not files:

        raise RuntimeError(
            f"No se encontraron imágenes en:\n"
            f"{SHIPSNET_DIR}"
        )


    for path in files:

        parts = (
            path.stem
            .split("__")
        )


        if len(parts) < 2:

            continue


        try:

            label = int(
                parts[0]
            )

        except Exception:

            continue


        if label not in [
            0,
            1
        ]:

            continue


        scene_id = parts[1]


        records.append({

            "path":
                path,

            "label":
                label,

            "scene_id":
                scene_id,

            "domain":
                "shipsnet",
        })


    return records


# ============================================================
# CARGAR DATASET EXTERNO
# ============================================================

def load_external_records(
    images_dir,
    csv_path,
    domain_name
):

    if not csv_path.exists():

        raise FileNotFoundError(
            f"No existe:\n"
            f"{csv_path}"
        )


    df = pd.read_csv(
        csv_path
    )


    required = {
        "filename",
        "label"
    }


    if not required.issubset(
        df.columns
    ):

        raise ValueError(
            f"{csv_path.name} debe contener "
            f"filename,label"
        )


    records = []


    for _, row in df.iterrows():

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


        if label not in [
            0,
            1
        ]:

            continue


        path = (
            images_dir
            /
            filename
        )


        if not path.exists():

            continue


        records.append({

            "path":
                path,

            "label":
                label,

            "scene_id":
                None,

            "domain":
                domain_name,
        })


    return records


# ============================================================
# SPLIT SHIPSNET POR SCENE ID
# ============================================================

def split_shipsnet(
    records
):

    labels = np.asarray([

        record[
            "label"
        ]

        for record
        in records

    ])


    groups = np.asarray([

        record[
            "scene_id"
        ]

        for record
        in records

    ])


    dummy = np.zeros(
        len(
            records
        )
    )


    splitter = (
        StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=SEED
        )
    )


    # ========================================================
    # Usamos el primer fold como holdout.
    #
    # Ningún scene_id aparece simultáneamente en train y valid.
    # ========================================================

    train_idx, valid_idx = next(

        splitter.split(
            dummy,
            labels,
            groups
        )
    )


    train_records = [

        records[
            index
        ]

        for index
        in train_idx
    ]


    valid_records = [

        records[
            index
        ]

        for index
        in valid_idx
    ]


    return (
        train_records,
        valid_records
    )


# ============================================================
# DATA LOADERS
# ============================================================

def create_loader(
    records,
    transform,
    shuffle
):

    dataset = ShipDataset(
        records,
        transform
    )


    loader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=shuffle,

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


    return loader


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(
    records
):

    labels = np.asarray([

        record[
            "label"
        ]

        for record
        in records

    ])


    counts = np.bincount(
        labels,
        minlength=2
    )


    total = counts.sum()


    weights = (

        total

        /

        (
            2.0
            *
            counts
        )
    )


    return torch.tensor(

        weights,

        dtype=torch.float32,

        device=DEVICE
    )


# ============================================================
# CREAR EFFICIENTNET
# ============================================================

def create_model():

    print(
        "\nCargando EfficientNet-B0 "
        "preentrenada en ImageNet..."
    )


    weights = (
        EfficientNet_B0_Weights.DEFAULT
    )


    model = efficientnet_b0(
        weights=weights
    )


    # ========================================================
    # EfficientNet-B0:
    #
    # classifier =
    # Sequential(
    #     Dropout(...)
    #     Linear(...)
    # )
    #
    # La última capa es classifier[1]
    # ========================================================

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
# FREEZE BACKBONE
# ============================================================

def freeze_backbone(
    model
):

    for parameter in (
        model
        .features
        .parameters()
    ):

        parameter.requires_grad = False


# ============================================================
# UNFREEZE TODO
# ============================================================

def unfreeze_all(
    model
):

    for parameter in (
        model.parameters()
    ):

        parameter.requires_grad = True


# ============================================================
# AMP CONTEXT
# ============================================================

def autocast_context():

    return torch.amp.autocast(

        device_type=DEVICE.type,

        enabled=(
            DEVICE.type
            ==
            "cuda"
        )
    )


# ============================================================
# ENTRENAR UNA ÉPOCA
# ============================================================

def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler
):

    model.train()


    running_loss = 0.0


    all_true = []

    all_pred = []


    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )


        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        # ====================================================
        # MIXED PRECISION
        # ====================================================

        with autocast_context():

            logits = model(
                images
            )

            loss = criterion(
                logits,
                labels
            )


        scaler.scale(
            loss
        ).backward()


        scaler.step(
            optimizer
        )


        scaler.update()


        running_loss += (

            loss.item()

            *
            images.size(0)
        )


        predictions = torch.argmax(

            logits,

            dim=1
        )


        all_true.extend(

            labels
            .detach()
            .cpu()
            .numpy()
        )


        all_pred.extend(

            predictions
            .detach()
            .cpu()
            .numpy()
        )


    epoch_loss = (

        running_loss

        /

        len(
            loader.dataset
        )
    )


    epoch_accuracy = (
        accuracy_score(
            all_true,
            all_pred
        )
    )


    return (
        epoch_loss,
        epoch_accuracy
    )


# ============================================================
# EVALUACIÓN
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader
):

    model.eval()


    all_true = []

    all_pred = []

    all_prob = []


    running_loss = 0.0


    criterion = nn.CrossEntropyLoss()


    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )


        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        with autocast_context():

            logits = model(
                images
            )


            loss = criterion(
                logits,
                labels
            )


        probabilities = torch.softmax(

            logits,

            dim=1

        )[:, 1]


        predictions = torch.argmax(

            logits,

            dim=1
        )


        running_loss += (

            loss.item()

            *
            images.size(0)
        )


        all_true.extend(

            labels
            .detach()
            .cpu()
            .numpy()
        )


        all_pred.extend(

            predictions
            .detach()
            .cpu()
            .numpy()
        )


        all_prob.extend(

            probabilities
            .detach()
            .cpu()
            .numpy()
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
        all_prob,
        dtype=np.float64
    )


    # ========================================================
    # LOSS
    # ========================================================

    loss = (

        running_loss

        /

        len(
            loader.dataset
        )
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


    try:

        auc = roc_auc_score(
            y_true,
            probabilities
        )

    except Exception:

        auc = np.nan


    cm = confusion_matrix(

        y_true,

        y_pred,

        labels=[
            0,
            1
        ]
    )


    return {

        "loss":
            loss,

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

        "y_true":
            y_true,

        "y_pred":
            y_pred,

        "probabilities":
            probabilities,
    }


# ============================================================
# MOSTRAR MÉTRICAS
# ============================================================

def print_metrics(
    name,
    metrics
):

    print("\n")
    print(
        "=" * 70
    )

    print(
        name
    )

    print(
        "=" * 70
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
# ENTRENAR ETAPA
# ============================================================

def run_stage(
    model,
    train_loader,
    shipsnet_valid_loader,
    external_valid_loader,
    criterion,
    learning_rate,
    epochs,
    stage_name,
    history,
    best_state
):

    # ========================================================
    # OPTIMIZADOR
    # ========================================================

    optimizer = torch.optim.AdamW(

        filter(
            lambda parameter:
                parameter.requires_grad,
            model.parameters()
        ),

        lr=learning_rate,

        weight_decay=WEIGHT_DECAY
    )


    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(

            optimizer,

            T_max=epochs
        )
    )


    # ========================================================
    # MIXED PRECISION
    # ========================================================

    scaler = torch.amp.GradScaler(

        DEVICE.type,

        enabled=(
            DEVICE.type
            ==
            "cuda"
        )
    )


    patience_counter = 0


    # ========================================================
    # ÉPOCAS
    # ========================================================

    for epoch in range(
        1,
        epochs + 1
    ):

        start = time.time()


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_loss, train_accuracy = (
            train_epoch(

                model,

                train_loader,

                criterion,

                optimizer,

                scaler
            )
        )


        # ----------------------------------------------------
        # SHIPSNET VALID
        # ----------------------------------------------------

        ships_metrics = evaluate(

            model,

            shipsnet_valid_loader
        )


        # ----------------------------------------------------
        # EXTERNAL VALID
        # ----------------------------------------------------

        external_metrics = evaluate(

            model,

            external_valid_loader
        )


        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step()


        # ====================================================
        # SCORE DE SELECCIÓN
        # ====================================================

        selection_score = (

            SHIPSNET_SELECTION_WEIGHT

            *

            ships_metrics[
                "accuracy"
            ]

            +

            EXTERNAL_SELECTION_WEIGHT

            *

            external_metrics[
                "balanced_accuracy"
            ]
        )


        seconds = (
            time.time()
            -
            start
        )


        current_lr = (
            optimizer
            .param_groups[0][
                "lr"
            ]
        )


        # ====================================================
        # PRINT
        # ====================================================

        print("\n")
        print(
            "=" * 75
        )

        print(
            f"{stage_name}"
            f" | Epoch "
            f"{epoch}/{epochs}"
        )

        print(
            "=" * 75
        )


        print(
            f"Train loss        : "
            f"{train_loss:.4f}"
        )


        print(
            f"Train accuracy    : "
            f"{train_accuracy * 100:.2f}%"
        )


        print(
            f"Learning rate     : "
            f"{current_lr:.8f}"
        )


        print(
            "\nShipsNet validation:"
        )


        print(
            f"  Accuracy        : "
            f"{ships_metrics['accuracy'] * 100:.2f}%"
        )


        print(
            f"  F1              : "
            f"{ships_metrics['f1'] * 100:.2f}%"
        )


        print(
            f"  Balanced Acc    : "
            f"{ships_metrics['balanced_accuracy'] * 100:.2f}%"
        )


        print(
            "\nExternal validation:"
        )


        print(
            f"  Accuracy        : "
            f"{external_metrics['accuracy'] * 100:.2f}%"
        )


        print(
            f"  F1              : "
            f"{external_metrics['f1'] * 100:.2f}%"
        )


        print(
            f"  Balanced Acc    : "
            f"{external_metrics['balanced_accuracy'] * 100:.2f}%"
        )


        print(
            f"\nSelection score   : "
            f"{selection_score * 100:.3f}%"
        )


        print(
            f"Tiempo            : "
            f"{seconds:.1f}s"
        )


        # ====================================================
        # HISTORIAL
        # ====================================================

        history.append({

            "stage":
                stage_name,

            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "train_accuracy":
                train_accuracy,

            "shipsnet_val_loss":
                ships_metrics[
                    "loss"
                ],

            "shipsnet_val_accuracy":
                ships_metrics[
                    "accuracy"
                ],

            "shipsnet_val_precision":
                ships_metrics[
                    "precision"
                ],

            "shipsnet_val_recall":
                ships_metrics[
                    "recall"
                ],

            "shipsnet_val_f1":
                ships_metrics[
                    "f1"
                ],

            "shipsnet_val_balanced":
                ships_metrics[
                    "balanced_accuracy"
                ],

            "shipsnet_val_auc":
                ships_metrics[
                    "roc_auc"
                ],

            "external_val_loss":
                external_metrics[
                    "loss"
                ],

            "external_val_accuracy":
                external_metrics[
                    "accuracy"
                ],

            "external_val_precision":
                external_metrics[
                    "precision"
                ],

            "external_val_recall":
                external_metrics[
                    "recall"
                ],

            "external_val_f1":
                external_metrics[
                    "f1"
                ],

            "external_val_balanced":
                external_metrics[
                    "balanced_accuracy"
                ],

            "external_val_auc":
                external_metrics[
                    "roc_auc"
                ],

            "selection_score":
                selection_score,

            "learning_rate":
                current_lr,

            "seconds":
                seconds,
        })


        # ====================================================
        # MEJOR MODELO COMBINADO
        # ====================================================

        if (
            selection_score
            >
            best_state[
                "score"
            ]
        ):

            best_state[
                "score"
            ] = selection_score


            best_state[
                "ships_accuracy"
            ] = ships_metrics[
                "accuracy"
            ]


            best_state[
                "external_balanced"
            ] = external_metrics[
                "balanced_accuracy"
            ]


            best_state[
                "stage"
            ] = stage_name


            best_state[
                "epoch"
            ] = epoch


            torch.save(

                model.state_dict(),

                BEST_MODEL_PATH
            )


            patience_counter = 0


            print(
                "\n✓ NUEVO MEJOR MODELO COMBINADO"
            )


        else:

            patience_counter += 1


        # ====================================================
        # MEJOR MODELO SHIPSNET
        # ====================================================

        if (
            ships_metrics[
                "accuracy"
            ]
            >
            best_state[
                "best_shipsnet_accuracy"
            ]
        ):

            best_state[
                "best_shipsnet_accuracy"
            ] = ships_metrics[
                "accuracy"
            ]


            best_state[
                "best_shipsnet_stage"
            ] = stage_name


            best_state[
                "best_shipsnet_epoch"
            ] = epoch


            torch.save(

                model.state_dict(),

                BEST_SHIPSNET_MODEL_PATH
            )


            print(
                "\n★ NUEVO RÉCORD SHIPSNET: "
                f"{ships_metrics['accuracy'] * 100:.3f}%"
            )


        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if (
            patience_counter
            >=
            PATIENCE
        ):

            print("\n")
            print(
                "=" * 75
            )

            print(
                "EARLY STOPPING"
            )

            print(
                "=" * 75
            )

            break


    return best_state


# ============================================================
# GUARDAR PREDICCIONES
# ============================================================

def save_predictions(
    metrics,
    filename
):

    df = pd.DataFrame({

        "real_label":
            metrics[
                "y_true"
            ],

        "predicted_label":
            metrics[
                "y_pred"
            ],

        "prob_ship":
            metrics[
                "probabilities"
            ],

        "correct":
            (
                metrics[
                    "y_true"
                ]
                ==
                metrics[
                    "y_pred"
                ]
            )
    })


    df.to_csv(

        RESULTS_DIR
        /
        filename,

        index=False
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "EFFICIENTNET-B0 - SHIP CLASSIFIER"
    )

    print(
        "=" * 80
    )


    # ========================================================
    # GPU
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


        print(
            f"CUDA PyTorch: "
            f"{torch.version.cuda}"
        )


        gpu_memory = (

            torch.cuda.get_device_properties(
                0
            ).total_memory

            /

            1024**3
        )


        print(
            f"VRAM: "
            f"{gpu_memory:.2f} GB"
        )


    else:

        print(
            "\n⚠ CUDA NO DETECTADA"
        )


    # ========================================================
    # SHIPSNET
    # ========================================================

    shipsnet_records = (
        load_shipsnet_records()
    )


    (
        shipsnet_train,
        shipsnet_valid
    ) = split_shipsnet(
        shipsnet_records
    )


    # ========================================================
    # EXTERNAL
    # ========================================================

    external_train = (
        load_external_records(

            EXTERNAL_TRAIN_DIR,

            EXTERNAL_TRAIN_CSV,

            "external_train"
        )
    )


    external_valid = (
        load_external_records(

            EXTERNAL_VALID_DIR,

            EXTERNAL_VALID_CSV,

            "external_valid"
        )
    )


    # ========================================================
    # TRAIN COMBINADO
    # ========================================================

    train_records = (

        shipsnet_train

        +

        external_train
    )


    # ========================================================
    # RESUMEN DATASET
    # ========================================================

    print("\n")
    print(
        "=" * 80
    )

    print(
        "DATASETS"
    )

    print(
        "=" * 80
    )


    print(
        f"ShipsNet total      : "
        f"{len(shipsnet_records)}"
    )


    print(
        f"ShipsNet train      : "
        f"{len(shipsnet_train)}"
    )


    print(
        f"ShipsNet group-val  : "
        f"{len(shipsnet_valid)}"
    )


    print(
        f"External train      : "
        f"{len(external_train)}"
    )


    print(
        f"External valid      : "
        f"{len(external_valid)}"
    )


    print(
        f"Train combinado     : "
        f"{len(train_records)}"
    )


    print(
        f"Input size          : "
        f"{IMG_SIZE}x{IMG_SIZE}"
    )


    print(
        f"Batch size          : "
        f"{BATCH_SIZE}"
    )


    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = create_loader(

        train_records,

        train_transform,

        True
    )


    shipsnet_valid_loader = (
        create_loader(

            shipsnet_valid,

            valid_transform,

            False
        )
    )


    external_valid_loader = (
        create_loader(

            external_valid,

            valid_transform,

            False
        )
    )


    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    class_weights = (
        calculate_class_weights(
            train_records
        )
    )


    print(
        f"\nClass weights: "
        f"{class_weights.detach().cpu().numpy()}"
    )


    # ========================================================
    # LOSS
    # ========================================================

    criterion = nn.CrossEntropyLoss(

        weight=class_weights,

        label_smoothing=LABEL_SMOOTHING
    )


    # ========================================================
    # MODELO
    # ========================================================

    model = create_model()


    model = model.to(
        DEVICE
    )


    # ========================================================
    # HISTORIAL
    # ========================================================

    history = []


    best_state = {

        "score":
            -1.0,

        "ships_accuracy":
            0.0,

        "external_balanced":
            0.0,

        "best_shipsnet_accuracy":
            0.0,

        "stage":
            None,

        "epoch":
            None,

        "best_shipsnet_stage":
            None,

        "best_shipsnet_epoch":
            None,
    }


    # ========================================================
    # ETAPA 1
    # ENTRENAR SOLO CLASSIFIER
    # ========================================================

    print("\n")
    print(
        "=" * 80
    )

    print(
        "ETAPA 1 - CLASSIFIER HEAD"
    )

    print(
        "=" * 80
    )


    freeze_backbone(
        model
    )


    best_state = run_stage(

        model=model,

        train_loader=train_loader,

        shipsnet_valid_loader=(
            shipsnet_valid_loader
        ),

        external_valid_loader=(
            external_valid_loader
        ),

        criterion=criterion,

        learning_rate=HEAD_LR,

        epochs=HEAD_EPOCHS,

        stage_name="HEAD",

        history=history,

        best_state=best_state
    )


    # ========================================================
    # ETAPA 2
    # FINE TUNING COMPLETO
    # ========================================================

    print("\n")
    print(
        "=" * 80
    )

    print(
        "ETAPA 2 - FINE TUNING"
    )

    print(
        "=" * 80
    )


    unfreeze_all(
        model
    )


    best_state = run_stage(

        model=model,

        train_loader=train_loader,

        shipsnet_valid_loader=(
            shipsnet_valid_loader
        ),

        external_valid_loader=(
            external_valid_loader
        ),

        criterion=criterion,

        learning_rate=FINETUNE_LR,

        epochs=FINETUNE_EPOCHS,

        stage_name="FINETUNE",

        history=history,

        best_state=best_state
    )


    # ========================================================
    # GUARDAR HISTORIAL
    # ========================================================

    history_df = pd.DataFrame(
        history
    )


    history_df.to_csv(

        RESULTS_DIR
        /
        "efficientnet_history.csv",

        index=False
    )


    # ========================================================
    # CARGAR MEJOR MODELO COMBINADO
    # ========================================================

    print("\n")
    print(
        "=" * 80
    )

    print(
        "CARGANDO MEJOR MODELO COMBINADO"
    )

    print(
        "=" * 80
    )


    state_dict = torch.load(

        BEST_MODEL_PATH,

        map_location=DEVICE,

        weights_only=True
    )


    model.load_state_dict(
        state_dict
    )


    model.eval()


    # ========================================================
    # EVALUACIÓN FINAL
    # ========================================================

    ships_metrics = evaluate(

        model,

        shipsnet_valid_loader
    )


    external_metrics = evaluate(

        model,

        external_valid_loader
    )


    # ========================================================
    # PRINT
    # ========================================================

    print_metrics(

        "SHIPSNET GROUP VALIDATION",

        ships_metrics
    )


    print_metrics(

        "EXTERNAL VALIDATION",

        external_metrics
    )


    # ========================================================
    # GUARDAR PREDICCIONES
    # ========================================================

    save_predictions(

        ships_metrics,

        "efficientnet_shipsnet_validation_predictions.csv"
    )


    save_predictions(

        external_metrics,

        "efficientnet_external_validation_predictions.csv"
    )


    # ========================================================
    # ERRORES SHIPSNET
    # ========================================================

    ships_errors = pd.DataFrame({

        "real_label":
            ships_metrics[
                "y_true"
            ],

        "predicted_label":
            ships_metrics[
                "y_pred"
            ],

        "prob_ship":
            ships_metrics[
                "probabilities"
            ],
    })


    ships_errors = ships_errors[

        ships_errors[
            "real_label"
        ]

        !=

        ships_errors[
            "predicted_label"
        ]
    ]


    ships_errors.to_csv(

        RESULTS_DIR
        /
        "efficientnet_shipsnet_errors.csv",

        index=False
    )


    # ========================================================
    # ERRORES EXTERNAL
    # ========================================================

    external_errors = pd.DataFrame({

        "real_label":
            external_metrics[
                "y_true"
            ],

        "predicted_label":
            external_metrics[
                "y_pred"
            ],

        "prob_ship":
            external_metrics[
                "probabilities"
            ],
    })


    external_errors = external_errors[

        external_errors[
            "real_label"
        ]

        !=

        external_errors[
            "predicted_label"
        ]
    ]


    external_errors.to_csv(

        RESULTS_DIR
        /
        "efficientnet_external_errors.csv",

        index=False
    )


    # ========================================================
    # JSON RESUMEN
    # ========================================================

    summary = {

        "architecture":
            "EfficientNet-B0",

        "input_size":
            IMG_SIZE,

        "batch_size":
            BATCH_SIZE,

        "best_combined": {

            "stage":
                best_state[
                    "stage"
                ],

            "epoch":
                best_state[
                    "epoch"
                ],

            "selection_score":
                float(
                    best_state[
                        "score"
                    ]
                ),
        },

        "best_shipsnet": {

            "stage":
                best_state[
                    "best_shipsnet_stage"
                ],

            "epoch":
                best_state[
                    "best_shipsnet_epoch"
                ],

            "accuracy":
                float(
                    best_state[
                        "best_shipsnet_accuracy"
                    ]
                ),
        },

        "shipsnet_validation": {

            "accuracy":
                float(
                    ships_metrics[
                        "accuracy"
                    ]
                ),

            "precision":
                float(
                    ships_metrics[
                        "precision"
                    ]
                ),

            "recall":
                float(
                    ships_metrics[
                        "recall"
                    ]
                ),

            "f1":
                float(
                    ships_metrics[
                        "f1"
                    ]
                ),

            "balanced_accuracy":
                float(
                    ships_metrics[
                        "balanced_accuracy"
                    ]
                ),

            "roc_auc":
                float(
                    ships_metrics[
                        "roc_auc"
                    ]
                ),

            "TN":
                int(
                    ships_metrics[
                        "TN"
                    ]
                ),

            "FP":
                int(
                    ships_metrics[
                        "FP"
                    ]
                ),

            "FN":
                int(
                    ships_metrics[
                        "FN"
                    ]
                ),

            "TP":
                int(
                    ships_metrics[
                        "TP"
                    ]
                ),
        },

        "external_validation": {

            "accuracy":
                float(
                    external_metrics[
                        "accuracy"
                    ]
                ),

            "precision":
                float(
                    external_metrics[
                        "precision"
                    ]
                ),

            "recall":
                float(
                    external_metrics[
                        "recall"
                    ]
                ),

            "f1":
                float(
                    external_metrics[
                        "f1"
                    ]
                ),

            "balanced_accuracy":
                float(
                    external_metrics[
                        "balanced_accuracy"
                    ]
                ),

            "roc_auc":
                float(
                    external_metrics[
                        "roc_auc"
                    ]
                ),

            "TN":
                int(
                    external_metrics[
                        "TN"
                    ]
                ),

            "FP":
                int(
                    external_metrics[
                        "FP"
                    ]
                ),

            "FN":
                int(
                    external_metrics[
                        "FN"
                    ]
                ),

            "TP":
                int(
                    external_metrics[
                        "TP"
                    ]
                ),
        }
    }


    with open(

        RESULTS_DIR
        /
        "efficientnet_summary.json",

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            summary,

            file,

            indent=4
        )


    # ========================================================
    # CSV RESUMEN
    # ========================================================

    summary_df = pd.DataFrame([{

        "architecture":
            "EfficientNet-B0",

        "shipsnet_accuracy":
            ships_metrics[
                "accuracy"
            ],

        "shipsnet_precision":
            ships_metrics[
                "precision"
            ],

        "shipsnet_recall":
            ships_metrics[
                "recall"
            ],

        "shipsnet_f1":
            ships_metrics[
                "f1"
            ],

        "shipsnet_balanced":
            ships_metrics[
                "balanced_accuracy"
            ],

        "shipsnet_auc":
            ships_metrics[
                "roc_auc"
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

        "external_balanced":
            external_metrics[
                "balanced_accuracy"
            ],

        "external_auc":
            external_metrics[
                "roc_auc"
            ],

        "best_shipsnet_accuracy_observed":
            best_state[
                "best_shipsnet_accuracy"
            ],

        "best_stage":
            best_state[
                "stage"
            ],

        "best_epoch":
            best_state[
                "epoch"
            ],
    }])


    summary_df.to_csv(

        RESULTS_DIR
        /
        "efficientnet_summary.csv",

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
        "RESULTADO FINAL"
    )

    print(
        "=" * 80
    )


    print(
        f"\nMejor modelo combinado:\n"
        f"{BEST_MODEL_PATH}"
    )


    print(
        f"\nMejor modelo ShipsNet:\n"
        f"{BEST_SHIPSNET_MODEL_PATH}"
    )


    print(
        f"\nMejor accuracy ShipsNet observada: "
        f"{best_state['best_shipsnet_accuracy'] * 100:.3f}%"
    )


    print(
        f"Mejor epoch combinado: "
        f"{best_state['stage']} "
        f"{best_state['epoch']}"
    )


    print(
        f"Mejor epoch ShipsNet: "
        f"{best_state['best_shipsnet_stage']} "
        f"{best_state['best_shipsnet_epoch']}"
    )


    print("\n")


    if (
        ships_metrics[
            "accuracy"
        ]
        >=
        0.99
    ):

        print(
            "✓ EFFICIENTNET SUPERA 99% "
            "EN SHIPSNET"
        )

    else:

        print(
            "EfficientNet no supera 99% "
            "en el checkpoint combinado."
        )


    if (
        best_state[
            "best_shipsnet_accuracy"
        ]
        >
        0.9937
    ):

        print(
            "★ EFFICIENTNET SUPERÓ EL "
            "99.37% DE MOBILENET"
        )

    else:

        print(
            "MobileNet sigue teniendo "
            "el mejor récord individual."
        )


    print(
        f"\nResultados guardados en:\n"
        f"{RESULTS_DIR}"
    )


    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()