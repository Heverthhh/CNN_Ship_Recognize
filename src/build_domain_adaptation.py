from pathlib import Path
import zipfile
import random
import shutil
import math

import cv2
import numpy as np
import pandas as pd


# ============================================================
# RUTAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ZIP_CANDIDATES = [
    BASE_DIR / "ships_aerial_images.zip",
    BASE_DIR / "archive(1).zip",
]

OUTPUT_ROOT = (
    BASE_DIR
    / "domain_adaptation"
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

FINAL_SIZE = 80

RANDOM_SEED = 42

TARGETS = {
    "train": {
        "ship": 1000,
        "no_ship": 1000,
    },
    "valid": {
        "ship": 300,
        "no_ship": 300,
    }
}


# ============================================================
# FILTROS DE CALIDAD
# ============================================================

MIN_BBOX_DIMENSION = 8.0

MAX_ASPECT_RATIO = 12.0

TARGET_OCCUPANCY = 0.60

MIN_OCCUPANCY = 0.50
MAX_OCCUPANCY = 0.70

MIN_SOURCE_CROP = 24

MAX_NEGATIVE_ATTEMPTS = 150

NEGATIVE_MARGIN_RATIO = 0.10


# ============================================================
# UTILIDADES
# ============================================================

def find_zip():

    for path in ZIP_CANDIDATES:

        if path.exists():
            return path

    raise FileNotFoundError(
        "No encontré el ZIP del dataset externo."
    )


def decode_image(data):

    array = np.frombuffer(
        data,
        dtype=np.uint8
    )

    return cv2.imdecode(
        array,
        cv2.IMREAD_COLOR
    )


# ============================================================
# LEER YOLO
# ============================================================

def parse_yolo_labels(
    text,
    width,
    height
):

    boxes = []

    if not text.strip():
        return boxes

    for line in text.splitlines():

        parts = line.strip().split()

        if len(parts) < 5:
            continue

        try:

            class_id = int(
                float(parts[0])
            )

            xc = float(parts[1])
            yc = float(parts[2])

            bw = float(parts[3])
            bh = float(parts[4])

        except ValueError:
            continue

        if class_id != 0:
            continue

        if not (
            0 <= xc <= 1
            and
            0 <= yc <= 1
            and
            0 < bw <= 1
            and
            0 < bh <= 1
        ):
            continue

        cx = xc * width
        cy = yc * height

        box_w = bw * width
        box_h = bh * height

        x1 = max(
            0,
            cx - box_w / 2
        )

        y1 = max(
            0,
            cy - box_h / 2
        )

        x2 = min(
            width,
            cx + box_w / 2
        )

        y2 = min(
            height,
            cy + box_h / 2
        )

        real_w = x2 - x1
        real_h = y2 - y1

        if (
            real_w <= 0
            or
            real_h <= 0
        ):
            continue

        boxes.append({

            "cx":
                (x1 + x2) / 2,

            "cy":
                (y1 + y2) / 2,

            "width":
                real_w,

            "height":
                real_h,

            "x1":
                x1,

            "y1":
                y1,

            "x2":
                x2,

            "y2":
                y2,
        })

    return boxes


# ============================================================
# VALIDAR BARCO
# ============================================================

def valid_ship_box(
    box,
    image_w,
    image_h
):

    bw = box["width"]
    bh = box["height"]

    if (
        bw < MIN_BBOX_DIMENSION
        or
        bh < MIN_BBOX_DIMENSION
    ):
        return False

    long_side = max(
        bw,
        bh
    )

    short_side = min(
        bw,
        bh
    )

    ratio = (
        long_side
        /
        short_side
    )

    if ratio > MAX_ASPECT_RATIO:
        return False

    crop_size = (
        long_side
        /
        TARGET_OCCUPANCY
    )

    if crop_size > min(
        image_w,
        image_h
    ):
        return False

    return True


# ============================================================
# CROP POSITIVO
# ============================================================

def ship_crop(
    image,
    box
):

    h, w = image.shape[:2]

    if not valid_ship_box(
        box,
        w,
        h
    ):
        return None, None

    long_side = max(
        box["width"],
        box["height"]
    )

    crop_size = int(
        math.ceil(
            long_side
            /
            TARGET_OCCUPANCY
        )
    )

    crop_size = max(
        crop_size,
        MIN_SOURCE_CROP
    )

    if (
        crop_size > w
        or
        crop_size > h
    ):
        return None, None

    x0 = int(
        round(
            box["cx"]
            -
            crop_size / 2
        )
    )

    y0 = int(
        round(
            box["cy"]
            -
            crop_size / 2
        )
    )

    x0 = max(
        0,
        min(
            x0,
            w - crop_size
        )
    )

    y0 = max(
        0,
        min(
            y0,
            h - crop_size
        )
    )

    x1 = x0 + crop_size
    y1 = y0 + crop_size

    # El barco completo debe caber
    if (
        box["x1"] < x0
        or
        box["y1"] < y0
        or
        box["x2"] > x1
        or
        box["y2"] > y1
    ):
        return None, None

    occupancy = (
        long_side
        /
        crop_size
    )

    if not (
        MIN_OCCUPANCY
        <= occupancy
        <= MAX_OCCUPANCY
    ):
        return None, None

    crop = image[
        y0:y1,
        x0:x1
    ]

    if crop.size == 0:
        return None, None

    crop = cv2.resize(
        crop,
        (
            FINAL_SIZE,
            FINAL_SIZE
        ),
        interpolation=cv2.INTER_AREA
    )

    info = {
        "crop_size":
            crop_size,

        "occupancy":
            occupancy,

        "bbox_width":
            box["width"],

        "bbox_height":
            box["height"],
    }

    return crop, info


# ============================================================
# INTERSECCIÓN
# ============================================================

def intersects(
    rect,
    box,
    margin
):

    x1, y1, x2, y2 = rect

    bx1 = box["x1"] - margin
    by1 = box["y1"] - margin

    bx2 = box["x2"] + margin
    by2 = box["y2"] + margin

    return not (
        x2 <= bx1
        or
        x1 >= bx2
        or
        y2 <= by1
        or
        y1 >= by2
    )


# ============================================================
# CROP NEGATIVO
# ============================================================

def negative_crop(
    image,
    boxes,
    crop_size,
    rng
):

    h, w = image.shape[:2]

    crop_size = int(
        crop_size
    )

    if (
        crop_size > w
        or
        crop_size > h
    ):
        return None

    margin = max(
        4,
        int(
            crop_size
            *
            NEGATIVE_MARGIN_RATIO
        )
    )

    for _ in range(
        MAX_NEGATIVE_ATTEMPTS
    ):

        x0 = rng.randint(
            0,
            w - crop_size
        )

        y0 = rng.randint(
            0,
            h - crop_size
        )

        x1 = x0 + crop_size
        y1 = y0 + crop_size

        rect = (
            x0,
            y0,
            x1,
            y1
        )

        collision = any(

            intersects(
                rect,
                box,
                margin
            )

            for box in boxes
        )

        if collision:
            continue

        crop = image[
            y0:y1,
            x0:x1
        ]

        if crop.size == 0:
            continue

        crop = cv2.resize(
            crop,
            (
                FINAL_SIZE,
                FINAL_SIZE
            ),
            interpolation=cv2.INTER_AREA
        )

        return crop

    return None


# ============================================================
# CARGAR SPLIT
# ============================================================

def load_split(
    zf,
    split
):

    all_names = zf.namelist()

    images = [

        name

        for name in all_names

        if (
            f"/{split}/images/"
            in name
            and
            name.lower().endswith(
                (
                    ".png",
                    ".jpg",
                    ".jpeg"
                )
            )
        )
    ]

    dataset = []

    print(
        f"\nLeyendo {split}: "
        f"{len(images)} imágenes"
    )

    for i, image_member in enumerate(
        images
    ):

        try:

            data = zf.read(
                image_member
            )

            image = decode_image(
                data
            )

        except Exception:
            continue

        if image is None:
            continue

        h, w = image.shape[:2]

        label_member = (
            image_member
            .replace(
                "/images/",
                "/labels/"
            )
        )

        label_member = str(
            Path(
                label_member
            ).with_suffix(
                ".txt"
            )
        ).replace(
            "\\",
            "/"
        )

        try:

            label_text = (
                zf.read(
                    label_member
                )
                .decode(
                    "utf-8"
                )
            )

        except KeyError:

            label_text = ""

        boxes = parse_yolo_labels(
            label_text,
            w,
            h
        )

        dataset.append({

            "source":
                image_member,

            "image":
                image,

            "boxes":
                boxes,
        })

        if (
            (i + 1)
            %
            500
            ==
            0
        ):

            print(
                f"  {i + 1}"
                f"/"
                f"{len(images)}"
            )

    return dataset


# ============================================================
# GENERAR SPLIT BALANCEADO
# ============================================================

def generate_split(
    split,
    dataset,
    target_ship,
    target_no_ship,
    rng
):

    split_dir = (
        OUTPUT_ROOT
        /
        split
    )

    images_dir = (
        split_dir
        /
        "images"
    )

    images_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    records = []

    # ========================================================
    # CANDIDATOS POSITIVOS
    # ========================================================

    candidates_by_source = {}

    for index, item in enumerate(
        dataset
    ):

        candidates = []

        for box in item["boxes"]:

            crop, info = ship_crop(
                item["image"],
                box
            )

            if crop is None:
                continue

            candidates.append(
                (
                    crop,
                    info
                )
            )

        if candidates:

            candidates_by_source[
                index
            ] = candidates


    # ========================================================
    # PRIORIZAR UNA IMAGEN POR FUENTE
    # ========================================================

    source_indexes = list(
        candidates_by_source.keys()
    )

    rng.shuffle(
        source_indexes
    )

    selected_ships = []

    # Primera pasada
    for source_index in source_indexes:

        if (
            len(selected_ships)
            >=
            target_ship
        ):
            break

        candidates = (
            candidates_by_source[
                source_index
            ].copy()
        )

        rng.shuffle(
            candidates
        )

        crop, info = (
            candidates[0]
        )

        selected_ships.append(
            (
                source_index,
                crop,
                info
            )
        )


    # Segunda pasada si faltan
    if (
        len(selected_ships)
        <
        target_ship
    ):

        remaining = []

        for source_index, candidates in (
            candidates_by_source.items()
        ):

            for crop, info in candidates[1:]:

                remaining.append(
                    (
                        source_index,
                        crop,
                        info
                    )
                )

        rng.shuffle(
            remaining
        )

        for candidate in remaining:

            if (
                len(selected_ships)
                >=
                target_ship
            ):
                break

            selected_ships.append(
                candidate
            )


    # ========================================================
    # GUARDAR POSITIVOS
    # ========================================================

    positive_crop_sizes = []

    for i, (
        source_index,
        crop,
        info
    ) in enumerate(
        selected_ships,
        start=1
    ):

        filename = (
            f"{split}_ship_"
            f"{i:05d}.png"
        )

        cv2.imwrite(
            str(
                images_dir
                /
                filename
            ),
            crop
        )

        positive_crop_sizes.append(
            info[
                "crop_size"
            ]
        )

        records.append({

            "filename":
                filename,

            "label":
                1,

            "class_name":
                "ship",

            "source_image":
                Path(
                    dataset[
                        source_index
                    ][
                        "source"
                    ]
                ).name,

            "source_crop_size":
                info[
                    "crop_size"
                ],

            "occupancy":
                info[
                    "occupancy"
                ],

            "bbox_width":
                info[
                    "bbox_width"
                ],

            "bbox_height":
                info[
                    "bbox_height"
                ]
        })


    # ========================================================
    # NEGATIVOS
    # ========================================================

    negative_count = 0

    attempts = 0

    if not positive_crop_sizes:

        positive_crop_sizes = [
            FINAL_SIZE
        ]

    while (
        negative_count
        <
        target_no_ship
        and
        attempts
        <
        50000
    ):

        attempts += 1

        item_index = rng.randrange(
            len(dataset)
        )

        item = dataset[
            item_index
        ]

        crop_size = rng.choice(
            positive_crop_sizes
        )

        crop = negative_crop(
            item["image"],
            item["boxes"],
            crop_size,
            rng
        )

        if crop is None:
            continue

        negative_count += 1

        filename = (
            f"{split}_noship_"
            f"{negative_count:05d}.png"
        )

        cv2.imwrite(
            str(
                images_dir
                /
                filename
            ),
            crop
        )

        records.append({

            "filename":
                filename,

            "label":
                0,

            "class_name":
                "no_ship",

            "source_image":
                Path(
                    item[
                        "source"
                    ]
                ).name,

            "source_crop_size":
                crop_size,

            "occupancy":
                0,

            "bbox_width":
                None,

            "bbox_height":
                None
        })


    # ========================================================
    # CSV
    # ========================================================

    df = pd.DataFrame(
        records
    )

    df = df.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(
        drop=True
    )

    labels_path = (
        split_dir
        /
        "labels.csv"
    )

    df.to_csv(
        labels_path,
        index=False
    )

    print(
        f"\n{split.upper()}"
    )

    print(
        f"BARCO: "
        f"{(df['label'] == 1).sum()}"
    )

    print(
        f"NO BARCO: "
        f"{(df['label'] == 0).sum()}"
    )

    print(
        f"TOTAL: "
        f"{len(df)}"
    )

    print(
        f"Fuentes positivas distintas: "
        f"{df[df['label'] == 1]['source_image'].nunique()}"
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    rng = random.Random(
        RANDOM_SEED
    )

    zip_path = find_zip()

    print(
        "=" * 75
    )

    print(
        "GENERADOR PARA DOMAIN ADAPTATION"
    )

    print(
        "=" * 75
    )

    print(
        f"\nZIP:\n{zip_path}"
    )

    # --------------------------------------------------------
    # Limpiar salida
    # --------------------------------------------------------

    if OUTPUT_ROOT.exists():

        shutil.rmtree(
            OUTPUT_ROOT
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # GENERAR TRAIN Y VALID
    # ========================================================

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as zf:

        train_dataset = load_split(
            zf,
            "train"
        )

        valid_dataset = load_split(
            zf,
            "valid"
        )


        generate_split(
            split="train",
            dataset=train_dataset,
            target_ship=TARGETS[
                "train"
            ][
                "ship"
            ],
            target_no_ship=TARGETS[
                "train"
            ][
                "no_ship"
            ],
            rng=rng
        )


        generate_split(
            split="valid",
            dataset=valid_dataset,
            target_ship=TARGETS[
                "valid"
            ][
                "ship"
            ],
            target_no_ship=TARGETS[
                "valid"
            ][
                "no_ship"
            ],
            rng=rng
        )


    # ========================================================
    # FINAL
    # ========================================================

    print("\n")
    print(
        "=" * 75
    )

    print(
        "DOMAIN ADAPTATION DATASET CREADO"
    )

    print(
        "=" * 75
    )

    print(
        f"\n{OUTPUT_ROOT}"
    )

    print(
        "\nIMPORTANTE:"
    )

    print(
        "El split TEST externo NO fue utilizado."
    )

    print(
        "test_externo_balanceado_v3 permanece "
        "como evaluación final."
    )

    print(
        "=" * 75
    )


if __name__ == "__main__":

    main()