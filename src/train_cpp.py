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
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
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
# CONFIGURACIÓN
# ============================================================

SEED = 42

BASE_DIR = Path(__file__).resolve().parent.parent

SHIPSNET_DIR = BASE_DIR / "dataset" / "raw"

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
    / "cnn"
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


# ============================================================
# HIPERPARÁMETROS
# ============================================================

# La imagen original sigue siendo 80x80.
# Se aumenta internamente a 96x96 para MobileNet.
IMG_SIZE = 96

BATCH_SIZE = 64

HEAD_EPOCHS = 5
FINETUNE_EPOCHS = 35

HEAD_LR = 1e-3
FINETUNE_LR = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 4

PATIENCE = 9

LABEL_SMOOTHING = 0.05


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
# TRANSFORMACIONES
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


train_transform = transforms.Compose([

    transforms.RandomResizedCrop(
        IMG_SIZE,
        scale=(0.78, 1.0),
        ratio=(0.90, 1.10)
    ),

    # En vista aérea un barco puede estar orientado
    # en cualquier dirección
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

        if self.transform:

            image = self.transform(
                image
            )

        return (
            image,
            label
        )


# ============================================================
# SHIPSNET
# ============================================================

def load_shipsnet_records():

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
# DATASET EXTERNO
# ============================================================

def load_external_records(
    images_dir,
    csv_path,
    domain_name
):

    df = pd.read_csv(
        csv_path
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

    labels = np.array([
        r["label"]
        for r in records
    ])

    groups = np.array([
        r["scene_id"]
        for r in records
    ])

    dummy = np.zeros(
        len(records)
    )

    splitter = (
        StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=SEED
        )
    )

    # Tomamos fold 1 como validación fija.
    train_idx, valid_idx = next(
        splitter.split(
            dummy,
            labels,
            groups
        )
    )

    train_records = [
        records[i]
        for i in train_idx
    ]

    valid_records = [
        records[i]
        for i in valid_idx
    ]

    return (
        train_records,
        valid_records
    )


# ============================================================
# DATALOADER
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

    return DataLoader(

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
        ),
    )


# ============================================================
# MODELO
# ============================================================

def create_model():

    weights = (
        MobileNet_V3_Small_Weights.DEFAULT
    )

    model = mobilenet_v3_small(
        weights=weights
    )

    # Última capa
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
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(
    records
):

    labels = np.array([
        r["label"]
        for r in records
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
# TRAIN EPOCH
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


        with torch.cuda.amp.autocast(
            enabled=(
                DEVICE.type
                ==
                "cuda"
            )
        ):

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
            labels.cpu().numpy()
        )

        all_pred.extend(
            predictions.cpu().numpy()
        )

        all_prob.extend(
            probabilities.cpu().numpy()
        )


    y_true = np.array(
        all_true
    )

    y_pred = np.array(
        all_pred
    )

    probabilities = np.array(
        all_prob
    )


    loss = (
        running_loss
        /
        len(
            loader.dataset
        )
    )


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
        labels=[0, 1]
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
            int(cm[0, 0]),

        "FP":
            int(cm[0, 1]),

        "FN":
            int(cm[1, 0]),

        "TP":
            int(cm[1, 1]),

        "y_true":
            y_true,

        "y_pred":
            y_pred,

        "probabilities":
            probabilities,
    }


# ============================================================
# PRINT
# ============================================================

def print_metrics(
    name,
    metrics
):

    print(
        f"\n{name}"
    )

    print(
        f"Accuracy: "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Recall: "
        f"{metrics['recall'] * 100:.2f}%"
    )

    print(
        f"F1: "
        f"{metrics['f1'] * 100:.2f}%"
    )

    print(
        f"Balanced Acc: "
        f"{metrics['balanced_accuracy'] * 100:.2f}%"
    )

    print(
        f"ROC-AUC: "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        "Matriz:"
    )

    print([
        [
            metrics["TN"],
            metrics["FP"]
        ],
        [
            metrics["FN"],
            metrics["TP"]
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
    lr,
    epochs,
    stage_name,
    history,
    best_state
):

    optimizer = torch.optim.AdamW(

        filter(
            lambda p:
                p.requires_grad,
            model.parameters()
        ),

        lr=lr,

        weight_decay=WEIGHT_DECAY
    )


    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs
        )
    )


    scaler = torch.cuda.amp.GradScaler(
        enabled=(
            DEVICE.type
            ==
            "cuda"
        )
    )


    patience_counter = 0


    for epoch in range(
        1,
        epochs + 1
    ):

        start = time.time()


        train_loss, train_acc = (
            train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler
            )
        )


        ships_metrics = evaluate(
            model,
            shipsnet_valid_loader
        )


        external_metrics = evaluate(
            model,
            external_valid_loader
        )


        scheduler.step()


        # ----------------------------------------------------
        # CRITERIO DE SELECCIÓN
        #
        # Priorizamos ShipsNet porque el test oficial
        # probablemente está relacionado con ese dominio,
        # pero exigimos también robustez externa.
        # ----------------------------------------------------

        selection_score = (

            0.75
            *
            ships_metrics[
                "accuracy"
            ]

            +

            0.25
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
            f"Train loss : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train acc  : "
            f"{train_acc * 100:.2f}%"
        )


        print(
            f"\nShipsNet VAL:"
            f" {ships_metrics['accuracy'] * 100:.2f}%"
            f" | F1 "
            f"{ships_metrics['f1'] * 100:.2f}%"
        )


        print(
            f"External VAL:"
            f" {external_metrics['accuracy'] * 100:.2f}%"
            f" | Balanced "
            f"{external_metrics['balanced_accuracy'] * 100:.2f}%"
        )


        print(
            f"Selection score: "
            f"{selection_score * 100:.3f}%"
        )


        print(
            f"Tiempo: "
            f"{seconds:.1f}s"
        )


        history.append({

            "stage":
                stage_name,

            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "train_accuracy":
                train_acc,

            "shipsnet_val_loss":
                ships_metrics[
                    "loss"
                ],

            "shipsnet_val_accuracy":
                ships_metrics[
                    "accuracy"
                ],

            "shipsnet_val_f1":
                ships_metrics[
                    "f1"
                ],

            "shipsnet_val_balanced":
                ships_metrics[
                    "balanced_accuracy"
                ],

            "external_val_accuracy":
                external_metrics[
                    "accuracy"
                ],

            "external_val_f1":
                external_metrics[
                    "f1"
                ],

            "external_val_balanced":
                external_metrics[
                    "balanced_accuracy"
                ],

            "selection_score":
                selection_score,

            "seconds":
                seconds,
        })


        # ====================================================
        # MEJOR GENERAL
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

                MODELS_DIR
                /
                "cnn_mobilenetv3_best.pt"
            )


            patience_counter = 0


            print(
                "\n✓ NUEVO MEJOR MODELO"
            )


        else:

            patience_counter += 1


        # ====================================================
        # MEJOR SOLO SHIPSNET
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


            torch.save(

                model.state_dict(),

                MODELS_DIR
                /
                "cnn_mobilenetv3_best_shipsnet.pt"
            )


        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if (
            patience_counter
            >= PATIENCE
        ):

            print(
                "\nEarly stopping."
            )

            break


    return best_state


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "MOBILENETV3-SMALL - SHIP CLASSIFIER"
    )

    print(
        "=" * 80
    )


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
            f"CUDA: "
            f"{torch.version.cuda}"
        )

    else:

        print(
            "\n⚠ CUDA no detectada."
        )

        print(
            "Funcionará en CPU, "
            "pero será bastante más lento."
        )


    # ========================================================
    # DATOS
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
    # ENTRENAMIENTO COMBINADO
    # ========================================================

    train_records = (

        shipsnet_train

        +

        external_train
    )


    print("\n")
    print("=" * 80)
    print("DATASETS")
    print("=" * 80)


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


    # ========================================================
    # LOADERS
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
    # PESOS
    # ========================================================

    class_weights = (
        calculate_class_weights(
            train_records
        )
    )


    print(
        f"\nClass weights:"
        f" {class_weights.detach().cpu().numpy()}"
    )


    criterion = nn.CrossEntropyLoss(

        weight=class_weights,

        label_smoothing=LABEL_SMOOTHING
    )


    # ========================================================
    # MODELO
    # ========================================================

    print(
        "\nCargando MobileNetV3-Small "
        "preentrenada..."
    )


    model = create_model()

    model = model.to(
        DEVICE
    )


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
    }


    # ========================================================
    # ETAPA 1
    # SOLO CABEZA
    # ========================================================

    print("\n")
    print("=" * 80)
    print("ETAPA 1 - CLASSIFIER HEAD")
    print("=" * 80)


    freeze_backbone(
        model
    )


    best_state = run_stage(

        model,

        train_loader,

        shipsnet_valid_loader,

        external_valid_loader,

        criterion,

        HEAD_LR,

        HEAD_EPOCHS,

        "HEAD",

        history,

        best_state
    )


    # ========================================================
    # ETAPA 2
    # FINE-TUNING COMPLETO
    # ========================================================

    print("\n")
    print("=" * 80)
    print("ETAPA 2 - FINE TUNING")
    print("=" * 80)


    unfreeze_all(
        model
    )


    best_state = run_stage(

        model,

        train_loader,

        shipsnet_valid_loader,

        external_valid_loader,

        criterion,

        FINETUNE_LR,

        FINETUNE_EPOCHS,

        "FINETUNE",

        history,

        best_state
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
        "cnn_history.csv",

        index=False
    )


    # ========================================================
    # CARGAR MEJOR MODELO
    # ========================================================

    print("\n")
    print("=" * 80)
    print("CARGANDO MEJOR MODELO")
    print("=" * 80)


    model.load_state_dict(

        torch.load(

            MODELS_DIR
            /
            "cnn_mobilenetv3_best.pt",

            map_location=DEVICE
        )
    )


    model.eval()


    # ========================================================
    # RESULTADOS FINALES
    # ========================================================

    ships_metrics = evaluate(

        model,

        shipsnet_valid_loader
    )


    external_metrics = evaluate(

        model,

        external_valid_loader
    )


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

    ships_df = pd.DataFrame({

        "real":
            ships_metrics[
                "y_true"
            ],

        "prediction":
            ships_metrics[
                "y_pred"
            ],

        "prob_ship":
            ships_metrics[
                "probabilities"
            ],
    })


    ships_df.to_csv(

        RESULTS_DIR
        /
        "cnn_shipsnet_validation_predictions.csv",

        index=False
    )


    external_df = pd.DataFrame({

        "real":
            external_metrics[
                "y_true"
            ],

        "prediction":
            external_metrics[
                "y_pred"
            ],

        "prob_ship":
            external_metrics[
                "probabilities"
            ],
    })


    external_df.to_csv(

        RESULTS_DIR
        /
        "cnn_external_validation_predictions.csv",

        index=False
    )


    # ========================================================
    # RESUMEN
    # ========================================================

    summary = {

        "architecture":
            "MobileNetV3-Small",

        "input_size":
            IMG_SIZE,

        "best_stage":
            best_state[
                "stage"
            ],

        "best_epoch":
            best_state[
                "epoch"
            ],

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
        },
    }


    with open(

        RESULTS_DIR
        /
        "cnn_summary.json",

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )


    # ========================================================
    # FINAL
    # ========================================================

    print("\n")
    print("=" * 80)
    print("FINAL")
    print("=" * 80)


    print(
        f"Mejor modelo:\n"
        f"{MODELS_DIR / 'cnn_mobilenetv3_best.pt'}"
    )


    print(
        f"\nMejor accuracy ShipsNet observada:"
        f" "
        f"{best_state['best_shipsnet_accuracy'] * 100:.3f}%"
    )


    print(
        "\nMeta:"
    )

    if (
        ships_metrics[
            "accuracy"
        ]
        >=
        0.99
    ):

        print(
            "✓ SUPERAMOS 99% EN "
            "SHIPSNET GROUP VALIDATION"
        )

    else:

        print(
            "Aún no llegamos al 99%."
        )

        print(
            "El siguiente paso será "
            "CNN + HOG-SVM ensemble."
        )


    print("=" * 80)


if __name__ == "__main__":

    main()