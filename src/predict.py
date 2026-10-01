from pathlib import Path
import json
import time

import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.models import efficientnet_b0


# ============================================================
# RUTAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "ship_classifier_final.pt"
)

CONFIG_PATH = (
    BASE_DIR
    / "models"
    / "ship_classifier_final_config.json"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# GLOBALES
# ============================================================

_MODEL = None
_CONFIG = None
_TRANSFORM = None


# ============================================================
# ARQUITECTURA
# ============================================================

def create_model():

    model = efficientnet_b0(
        weights=None
    )

    in_features = (
        model.classifier[1]
        .in_features
    )

    model.classifier[1] = nn.Linear(
        in_features,
        2
    )

    return model


# ============================================================
# CARGAR MODELO
# ============================================================

def load_model(
    force_reload=False
):

    global _MODEL
    global _CONFIG
    global _TRANSFORM

    if (
        _MODEL is not None
        and
        not force_reload
    ):

        return _MODEL


    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"No existe el modelo:\n"
            f"{MODEL_PATH}"
        )


    if not CONFIG_PATH.exists():

        raise FileNotFoundError(
            f"No existe la configuración:\n"
            f"{CONFIG_PATH}"
        )


    # ========================================================
    # CONFIG
    # ========================================================

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        _CONFIG = json.load(
            file
        )


    input_size = int(
        _CONFIG.get(
            "input_size",
            128
        )
    )


    normalization = (
        _CONFIG.get(
            "normalization",
            {}
        )
    )


    mean = normalization.get(
        "mean",
        [
            0.485,
            0.456,
            0.406
        ]
    )


    std = normalization.get(
        "std",
        [
            0.229,
            0.224,
            0.225
        ]
    )


    # ========================================================
    # TRANSFORM
    # ========================================================

    _TRANSFORM = transforms.Compose([

        transforms.Resize(
            (
                input_size,
                input_size
            )
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=mean,
            std=std
        ),
    ])


    # ========================================================
    # MODELO
    # ========================================================

    model = create_model()


    try:

        state_dict = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True
        )

    except TypeError:

        # Compatibilidad con versiones antiguas
        # de PyTorch.
        state_dict = torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )


    model.load_state_dict(
        state_dict
    )


    model = model.to(
        DEVICE
    )


    model.eval()


    _MODEL = model


    # ========================================================
    # WARM-UP GPU
    # ========================================================

    if DEVICE.type == "cuda":

        dummy = torch.zeros(
            (
                1,
                3,
                input_size,
                input_size
            ),
            device=DEVICE
        )


        with torch.no_grad():

            _MODEL(
                dummy
            )


        torch.cuda.synchronize()


    return _MODEL


# ============================================================
# CONFIG
# ============================================================

def get_config():

    if _CONFIG is None:

        load_model()

    return _CONFIG


# ============================================================
# INFO DEL MODELO
# ============================================================

def get_model_info():

    config = get_config()

    validated = config.get(
        "validated_performance",
        {}
    )


    return {

        "architecture":
            config.get(
                "architecture",
                "EfficientNet-B0"
            ),

        "device":
            str(
                DEVICE
            ),

        "gpu":
            (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "CPU"
            ),

        "input_size":
            config.get(
                "input_size",
                128
            ),

        "threshold":
            config.get(
                "threshold",
                0.5
            ),

        "cv_accuracy":
            validated.get(
                "shipsnet_5fold_accuracy_mean",
                None
            ),

        "cv_std":
            validated.get(
                "shipsnet_5fold_accuracy_std",
                None
            ),

        "cv_recall":
            validated.get(
                "shipsnet_oof_recall",
                None
            ),

        "cv_auc":
            validated.get(
                "shipsnet_oof_roc_auc",
                None
            ),

        "external_tta_accuracy":
            validated.get(
                "external_effnet_tta_accuracy",
                None
            ),
    }


# ============================================================
# VARIANTES TTA
# ============================================================

def generate_tta_images(
    image
):

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
# PREPARAR BATCH
# ============================================================

def prepare_batch(
    image,
    use_tta
):

    if use_tta:

        variants = (
            generate_tta_images(
                image
            )
        )

    else:

        variants = [
            image
        ]


    tensors = [

        _TRANSFORM(
            variant
        )

        for variant
        in variants
    ]


    batch = torch.stack(
        tensors
    )


    batch = batch.to(
        DEVICE,
        non_blocking=True
    )


    return batch


# ============================================================
# PREDICCIÓN
# ============================================================

def predict_image(
    image_path,
    use_tta=False
):

    load_model()


    total_start = (
        time.perf_counter()
    )


    # ========================================================
    # ABRIR IMAGEN
    # ========================================================

    preprocessing_start = (
        time.perf_counter()
    )


    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )


    batch = prepare_batch(
        image,
        use_tta
    )


    if DEVICE.type == "cuda":

        torch.cuda.synchronize()


    preprocessing_end = (
        time.perf_counter()
    )


    # ========================================================
    # INFERENCIA
    # ========================================================

    inference_start = (
        time.perf_counter()
    )


    with torch.no_grad():

        logits = _MODEL(
            batch
        )


        probabilities = torch.softmax(
            logits,
            dim=1
        )[:, 1]


    if DEVICE.type == "cuda":

        torch.cuda.synchronize()


    inference_end = (
        time.perf_counter()
    )


    # ========================================================
    # PROBABILIDAD FINAL
    # ========================================================

    prob_ship = float(
        probabilities
        .mean()
        .item()
    )


    if len(
        probabilities
    ) > 1:

        tta_std = float(
            probabilities
            .std()
            .item()
        )

    else:

        tta_std = 0.0


    config = get_config()


    threshold = float(
        config.get(
            "threshold",
            0.5
        )
    )


    prediction = int(
        prob_ship
        >=
        threshold
    )


    class_name = (
        "BARCO"
        if prediction == 1
        else "NO BARCO"
    )


    prob_no_ship = (
        1.0
        -
        prob_ship
    )


    total_end = (
        time.perf_counter()
    )


    # ========================================================
    # TIEMPOS
    # ========================================================

    preprocessing_ms = (

        preprocessing_end

        -

        preprocessing_start

    ) * 1000.0


    inference_ms = (

        inference_end

        -

        inference_start

    ) * 1000.0


    total_ms = (

        total_end

        -

        total_start

    ) * 1000.0


    # ========================================================
    # SALIDA
    # ========================================================

    return {

        "prediction":
            prediction,

        "class_name":
            class_name,

        "prob_ship":
            prob_ship,

        "prob_no_ship":
            prob_no_ship,

        "threshold":
            threshold,

        "use_tta":
            bool(
                use_tta
            ),

        "tta_std":
            tta_std,

        "preprocessing_ms":
            preprocessing_ms,

        "inference_ms":
            inference_ms,

        "total_ms":
            total_ms,

        "device":
            str(
                DEVICE
            ),

        # ----------------------------------------------------
        # Compatibilidad con versiones antiguas de app.py
        # ----------------------------------------------------

        "decision_score":
            prob_ship,

        "feature_extraction_ms":
            preprocessing_ms,
    }


# ============================================================
# TEST DIRECTO
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 70
    )

    print(
        "SHIP CLASSIFIER - FINAL MODEL"
    )

    print(
        "=" * 70
    )


    load_model()


    info = get_model_info()


    print(
        f"\nArquitectura : "
        f"{info['architecture']}"
    )


    print(
        f"Device       : "
        f"{info['device']}"
    )


    print(
        f"GPU          : "
        f"{info['gpu']}"
    )


    print(
        f"Input        : "
        f"{info['input_size']}x"
        f"{info['input_size']}"
    )


    print(
        f"Threshold    : "
        f"{info['threshold']}"
    )


    if (
        info[
            "cv_accuracy"
        ]
        is not None
    ):

        print(
            f"\n5-fold CV    : "
            f"{info['cv_accuracy'] * 100:.3f}%"
        )


    if (
        info[
            "cv_recall"
        ]
        is not None
    ):

        print(
            f"Recall OOF   : "
            f"{info['cv_recall'] * 100:.2f}%"
        )


    print(
        "\n✓ MODELO CARGADO CORRECTAMENTE"
    )