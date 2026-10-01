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


# ============================================================
# CONFIGURACIÓN
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

MODELS_DIR = (
    BASE_DIR
    / "models"
)

RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "final_model"
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MODEL_PATH = (
    MODELS_DIR
    / "ship_classifier_final.pt"
)

CONFIG_PATH = (
    MODELS_DIR
    / "ship_classifier_final_config.json"
)


# ============================================================
# HIPERPARÁMETROS CONGELADOS
# ============================================================

IMG_SIZE = 128

BATCH_SIZE = 32

HEAD_EPOCHS = 5

# Derivado de los mejores epochs del 5-fold:
# 26, 26, 20, 12, 15
# mediana = 20
FINETUNE_EPOCHS = 20

HEAD_LR = 1e-3

FINETUNE_LR = 1e-4

WEIGHT_DECAY = 1e-4

LABEL_SMOOTHING = 0.05

NUM_WORKERS = 4


# ============================================================
# REPRODUCIBILIDAD
# ============================================================

def set_seed(seed):

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


# ============================================================
# AUGMENTATION FINAL
# ============================================================

train_transform = transforms.Compose([

    transforms.RandomResizedCrop(
        IMG_SIZE,
        scale=(0.75, 1.0),
        ratio=(0.88, 1.12)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomVerticalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=180
    ),

    transforms.ColorJitter(
        brightness=0.25,
        contrast=0.25,
        saturation=0.25,
        hue=0.05
    ),

    transforms.RandomApply(
        [
            transforms.GaussianBlur(
                kernel_size=3,
                sigma=(0.1, 1.0)
            )
        ],
        p=0.15
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

        image = Image.open(
            record["path"]
        ).convert(
            "RGB"
        )

        image = self.transform(
            image
        )

        label = int(
            record["label"]
        )

        return (
            image,
            label
        )


# ============================================================
# SHIPSNET
# ============================================================

def load_shipsnet():

    records = []

    files = sorted(
        SHIPSNET_DIR.glob(
            "*.png"
        )
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

        records.append({

            "path":
                path,

            "label":
                label,

            "domain":
                "shipsnet"
        })

    return records


# ============================================================
# EXTERNAL TRAIN
# ============================================================

def load_external():

    df = pd.read_csv(
        EXTERNAL_TRAIN_CSV
    )

    records = []

    for _, row in df.iterrows():

        filename = str(
            row["filename"]
        )

        label = int(
            row["label"]
        )

        path = (
            EXTERNAL_TRAIN_DIR
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

            "domain":
                "external"
        })

    return records


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(
    records
):

    labels = np.asarray([

        record["label"]

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
# MODELO
# ============================================================

def create_model():

    weights = (
        EfficientNet_B0_Weights.DEFAULT
    )


    model = efficientnet_b0(
        weights=weights
    )


    in_features = (
        model.classifier[1]
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
# FREEZE
# ============================================================

def freeze_backbone(
    model
):

    for parameter in (
        model.features.parameters()
    ):

        parameter.requires_grad = False


def unfreeze_all(
    model
):

    for parameter in (
        model.parameters()
    ):

        parameter.requires_grad = True


# ============================================================
# AMP
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
# ENTRENAR ÉPOCA
# ============================================================

def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler
):

    model.train()

    total_loss = 0.0

    correct = 0

    total = 0


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


        total_loss += (
            loss.item()
            *
            images.size(0)
        )


        predictions = torch.argmax(
            logits,
            dim=1
        )


        correct += (
            predictions
            ==
            labels
        ).sum().item()


        total += (
            labels.size(0)
        )


    epoch_loss = (
        total_loss
        /
        total
    )


    epoch_accuracy = (
        correct
        /
        total
    )


    return (
        epoch_loss,
        epoch_accuracy
    )


# ============================================================
# ETAPA DE ENTRENAMIENTO
# ============================================================

def train_stage(
    model,
    loader,
    criterion,
    learning_rate,
    epochs,
    stage_name,
    history
):

    optimizer = torch.optim.AdamW(

        filter(
            lambda parameter:
                parameter.requires_grad,
            model.parameters()
        ),

        lr=learning_rate,

        weight_decay=WEIGHT_DECAY
    )


    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(

            optimizer,

            T_max=epochs
        )
    )


    scaler = torch.amp.GradScaler(

        DEVICE.type,

        enabled=(
            DEVICE.type
            ==
            "cuda"
        )
    )


    for epoch in range(
        1,
        epochs + 1
    ):

        start = time.time()


        loss, accuracy = train_epoch(

            model,

            loader,

            criterion,

            optimizer,

            scaler
        )


        scheduler.step()


        elapsed = (
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


        print("\n")
        print(
            "=" * 70
        )

        print(
            f"{stage_name} "
            f"| Epoch "
            f"{epoch}/{epochs}"
        )

        print(
            "=" * 70
        )


        print(
            f"Loss      : "
            f"{loss:.4f}"
        )


        print(
            f"Accuracy  : "
            f"{accuracy * 100:.2f}%"
        )


        print(
            f"LR        : "
            f"{current_lr:.8f}"
        )


        print(
            f"Tiempo    : "
            f"{elapsed:.1f}s"
        )


        history.append({

            "stage":
                stage_name,

            "epoch":
                epoch,

            "loss":
                loss,

            "accuracy":
                accuracy,

            "learning_rate":
                current_lr,

            "seconds":
                elapsed,
        })


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "ENTRENAMIENTO FINAL - EFFICIENTNET-B0"
    )

    print(
        "=" * 80
    )


    print(
        f"\nDevice: {DEVICE}"
    )


    if DEVICE.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )


        print(
            f"CUDA: "
            f"{torch.version.cuda}"
        )


    # ========================================================
    # DATASET
    # ========================================================

    shipsnet = load_shipsnet()

    external = load_external()


    records = (

        shipsnet

        +

        external
    )


    random.shuffle(
        records
    )


    print("\n")
    print(
        "=" * 80
    )

    print(
        "DATASET FINAL"
    )

    print(
        "=" * 80
    )


    print(
        f"ShipsNet        : "
        f"{len(shipsnet)}"
    )


    print(
        f"External train  : "
        f"{len(external)}"
    )


    print(
        f"TOTAL           : "
        f"{len(records)}"
    )


    labels = np.asarray([

        record[
            "label"
        ]

        for record
        in records

    ])


    print(
        f"NO BARCO        : "
        f"{(labels == 0).sum()}"
    )


    print(
        f"BARCO           : "
        f"{(labels == 1).sum()}"
    )


    # ========================================================
    # DATALOADER
    # ========================================================

    dataset = ShipDataset(
        records,
        train_transform
    )


    loader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

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
    # CLASS WEIGHTS
    # ========================================================

    class_weights = (
        calculate_class_weights(
            records
        )
    )


    print(
        f"\nClass weights: "
        f"{class_weights.detach().cpu().numpy()}"
    )


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


    history = []


    # ========================================================
    # ETAPA 1
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


    train_stage(

        model=model,

        loader=loader,

        criterion=criterion,

        learning_rate=HEAD_LR,

        epochs=HEAD_EPOCHS,

        stage_name="HEAD",

        history=history
    )


    # ========================================================
    # ETAPA 2
    # ========================================================

    print("\n")
    print(
        "=" * 80
    )

    print(
        "ETAPA 2 - FINE TUNING COMPLETO"
    )

    print(
        "=" * 80
    )


    unfreeze_all(
        model
    )


    train_stage(

        model=model,

        loader=loader,

        criterion=criterion,

        learning_rate=FINETUNE_LR,

        epochs=FINETUNE_EPOCHS,

        stage_name="FINETUNE",

        history=history
    )


    # ========================================================
    # GUARDAR MODELO
    # ========================================================

    torch.save(

        model.state_dict(),

        MODEL_PATH
    )


    # ========================================================
    # HISTORIAL
    # ========================================================

    history_df = pd.DataFrame(
        history
    )


    history_df.to_csv(

        RESULTS_DIR
        /
        "final_training_history.csv",

        index=False
    )


    # ========================================================
    # CONFIG
    # ========================================================

    config = {

        "architecture":
            "EfficientNet-B0",

        "classes": {

            "0":
                "NO BARCO",

            "1":
                "BARCO",
        },

        "input_size":
            IMG_SIZE,

        "threshold":
            0.5,

        "normalization": {

            "mean":
                IMAGENET_MEAN,

            "std":
                IMAGENET_STD,
        },

        "training": {

            "shipsnet_samples":
                len(
                    shipsnet
                ),

            "external_samples":
                len(
                    external
                ),

            "total_samples":
                len(
                    records
                ),

            "head_epochs":
                HEAD_EPOCHS,

            "finetune_epochs":
                FINETUNE_EPOCHS,

            "head_lr":
                HEAD_LR,

            "finetune_lr":
                FINETUNE_LR,

            "weight_decay":
                WEIGHT_DECAY,

            "label_smoothing":
                LABEL_SMOOTHING,

            "seed":
                SEED,
        },

        "validated_performance": {

            "shipsnet_5fold_accuracy_mean":
                0.99750,

            "shipsnet_5fold_accuracy_std":
                0.00125,

            "shipsnet_oof_recall":
                1.0,

            "shipsnet_oof_roc_auc":
                0.9999535,

            "external_effnet_tta_accuracy":
                0.9625,
        },
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
    # FINAL
    # ========================================================

    print("\n")
    print(
        "=" * 80
    )

    print(
        "MODELO FINAL CREADO"
    )

    print(
        "=" * 80
    )


    print(
        f"\nModelo:\n"
        f"{MODEL_PATH}"
    )


    print(
        f"\nConfiguración:\n"
        f"{CONFIG_PATH}"
    )


    print(
        f"\nHistorial:\n"
        f"{RESULTS_DIR / 'final_training_history.csv'}"
    )


    print("\n")
    print(
        "Este modelo fue entrenado con "
        "todos los datos de desarrollo."
    )


    print(
        "NO se utilizó test_externo_final "
        "durante el entrenamiento."
    )


    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()