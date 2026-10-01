from pathlib import Path
import zipfile
import random
import shutil
import math

import cv2
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ZIP_PATH = (
    BASE_DIR
    / "ships_aerial_images.zip"
)

V3_LABELS = (
    BASE_DIR
    / "test_externo_balanceado_v3"
    / "labels.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "test_externo_final"
)

IMAGES_DIR = (
    OUTPUT_DIR
    / "images"
)

LABELS_PATH = (
    OUTPUT_DIR
    / "labels.csv"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "generation_report.csv"
)

FINAL_SIZE = 80

N_SHIP = 200
N_NO_SHIP = 200

RANDOM_SEED = 2026


# ============================================================
# MISMAS REGLAS DE V3
# NO SE MODIFICAN
# ============================================================

MIN_BBOX_DIMENSION = 8.0

MAX_ASPECT_RATIO = 12.0

TARGET_OCCUPANCY = 0.60

MIN_OCCUPANCY = 0.50
MAX_OCCUPANCY = 0.70

MIN_SOURCE_CROP = 24

MAX_NEGATIVE_ATTEMPTS = 200

NEGATIVE_MARGIN_RATIO = 0.10


# ============================================================
# DECODIFICAR
# ============================================================

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
# YOLO
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
            0.0,
            cx - box_w / 2
        )

        y1 = max(
            0.0,
            cy - box_h / 2
        )

        x2 = min(
            float(width),
            cx + box_w / 2
        )

        y2 = min(
            float(height),
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
# VALIDAR BBOX
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

    # BBox completo dentro del recorte
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

        "crop_x0":
            x0,

        "crop_y0":
            y0,

        "crop_x1":
            x1,

        "crop_y1":
            y1,
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
# NEGATIVO
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
        return None, None

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

        return crop, {

            "crop_size":
                crop_size,

            "crop_x0":
                x0,

            "crop_y0":
                y0,

            "crop_x1":
                x1,

            "crop_y1":
                y1,
        }

    return None, None


# ============================================================
# MAIN
# ============================================================

def main():

    rng = random.Random(
        RANDOM_SEED
    )

    print(
        "=" * 75
    )

    print(
        "TEST EXTERNO FINAL - FUENTES NO UTILIZADAS"
    )

    print(
        "=" * 75
    )


    # ========================================================
    # FUENTES YA USADAS EN V3
    # ========================================================

    if not V3_LABELS.exists():

        raise FileNotFoundError(
            f"No existe:\n{V3_LABELS}"
        )

    v3_df = pd.read_csv(
        V3_LABELS
    )

    if "source_image" not in v3_df.columns:

        raise ValueError(
            "El labels.csv de V3 debe contener "
            "la columna source_image."
        )

    used_sources = set(

        v3_df[
            "source_image"
        ]
        .dropna()
        .astype(str)
        .map(
            lambda x:
                Path(x).name
        )
    )


    print(
        f"\nFuentes utilizadas en V3: "
        f"{len(used_sources)}"
    )


    # ========================================================
    # PREPARAR SALIDA
    # ========================================================

    if OUTPUT_DIR.exists():

        shutil.rmtree(
            OUTPUT_DIR
        )

    IMAGES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # ZIP
    # ========================================================

    if not ZIP_PATH.exists():

        raise FileNotFoundError(
            f"No existe:\n{ZIP_PATH}"
        )


    records = []


    with zipfile.ZipFile(
        ZIP_PATH,
        "r"
    ) as zf:

        names = zf.namelist()

        test_images = [

            name

            for name in names

            if (
                "/test/images/"
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


        # ====================================================
        # EXCLUIR V3 POR IMAGEN FUENTE
        # ====================================================

        unused_images = [

            member

            for member in test_images

            if Path(
                member
            ).name
            not in
            used_sources
        ]


        print(
            f"Imágenes originales test: "
            f"{len(test_images)}"
        )

        print(
            f"Fuentes disponibles después "
            f"de excluir V3: "
            f"{len(unused_images)}"
        )


        rng.shuffle(
            unused_images
        )


        # ====================================================
        # CARGAR FUENTES
        # ====================================================

        dataset = []


        for i, member in enumerate(
            unused_images
        ):

            try:

                image = decode_image(
                    zf.read(
                        member
                    )
                )

            except Exception:
                continue


            if image is None:
                continue


            h, w = image.shape[:2]


            label_member = (
                member
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

                text = (
                    zf.read(
                        label_member
                    )
                    .decode(
                        "utf-8"
                    )
                )

            except KeyError:

                text = ""


            boxes = parse_yolo_labels(
                text,
                w,
                h
            )


            dataset.append({

                "source":
                    member,

                "image":
                    image,

                "boxes":
                    boxes,
            })


        print(
            f"Fuentes válidas cargadas: "
            f"{len(dataset)}"
        )


        # ====================================================
        # POSITIVOS
        # UNA FUENTE = MÁXIMO UN POSITIVO
        # ====================================================

        positive_candidates = []


        for source_index, item in enumerate(
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

                rng.shuffle(
                    candidates
                )

                crop, info = candidates[0]

                positive_candidates.append({

                    "source_index":
                        source_index,

                    "crop":
                        crop,

                    "info":
                        info,
                })


        rng.shuffle(
            positive_candidates
        )


        selected_positive = (
            positive_candidates[
                :N_SHIP
            ]
        )


        if len(
            selected_positive
        ) < N_SHIP:

            raise RuntimeError(
                "No existen suficientes "
                "positivos válidos."
            )


        positive_crop_sizes = []

        positive_source_indexes = set()


        for i, candidate in enumerate(
            selected_positive,
            start=1
        ):

            source_index = candidate[
                "source_index"
            ]

            item = dataset[
                source_index
            ]

            info = candidate[
                "info"
            ]

            filename = (
                f"final_ship_"
                f"{i:04d}.png"
            )


            cv2.imwrite(

                str(
                    IMAGES_DIR
                    /
                    filename
                ),

                candidate[
                    "crop"
                ]
            )


            positive_source_indexes.add(
                source_index
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
                        item[
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
                    ],

                "crop_x0":
                    info[
                        "crop_x0"
                    ],

                "crop_y0":
                    info[
                        "crop_y0"
                    ],

                "crop_x1":
                    info[
                        "crop_x1"
                    ],

                "crop_y1":
                    info[
                        "crop_y1"
                    ],
            })


        # ====================================================
        # NEGATIVOS
        #
        # Intentamos usar además fuentes distintas
        # de las utilizadas como positivos.
        # ====================================================

        negative_candidates = [

            index

            for index in range(
                len(dataset)
            )

            if index not in
            positive_source_indexes
        ]


        rng.shuffle(
            negative_candidates
        )


        negative_count = 0

        used_negative_sources = set()


        # Primera pasada: una negativa por fuente
        for source_index in (
            negative_candidates
        ):

            if (
                negative_count
                >= N_NO_SHIP
            ):
                break


            item = dataset[
                source_index
            ]


            crop_size = rng.choice(
                positive_crop_sizes
            )


            crop, info = negative_crop(

                item[
                    "image"
                ],

                item[
                    "boxes"
                ],

                crop_size,

                rng
            )


            if crop is None:
                continue


            negative_count += 1


            filename = (
                f"final_noship_"
                f"{negative_count:04d}.png"
            )


            cv2.imwrite(

                str(
                    IMAGES_DIR
                    /
                    filename
                ),

                crop
            )


            used_negative_sources.add(
                source_index
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
                    info[
                        "crop_size"
                    ],

                "occupancy":
                    0.0,

                "bbox_width":
                    None,

                "bbox_height":
                    None,

                "crop_x0":
                    info[
                        "crop_x0"
                    ],

                "crop_y0":
                    info[
                        "crop_y0"
                    ],

                "crop_x1":
                    info[
                        "crop_x1"
                    ],

                "crop_y1":
                    info[
                        "crop_y1"
                    ],
            })


        if negative_count < N_NO_SHIP:

            raise RuntimeError(
                f"Solo fue posible generar "
                f"{negative_count} negativos."
            )


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


    df.to_csv(
        LABELS_PATH,
        index=False
    )


    df.to_csv(
        REPORT_PATH,
        index=False
    )


    # ========================================================
    # VERIFICACIONES
    # ========================================================

    final_sources = set(
        df[
            "source_image"
        ].astype(str)
    )


    overlap = (
        final_sources
        &
        used_sources
    )


    print("\n")
    print(
        "=" * 75
    )

    print(
        "TEST FINAL GENERADO"
    )

    print(
        "=" * 75
    )


    print(
        f"\nBARCO: "
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
        f"\nFuentes distintas: "
        f"{df['source_image'].nunique()}"
    )


    print(
        f"Overlap de fuentes con V3: "
        f"{len(overlap)}"
    )


    if len(overlap) != 0:

        raise RuntimeError(
            "ERROR: existe contaminación "
            "entre V3 y test final."
        )


    print(
        "\n✓ NO EXISTE SUPERPOSICIÓN "
        "DE FUENTES CON V3"
    )


    print(
        f"\nImágenes:\n"
        f"{IMAGES_DIR}"
    )


    print(
        f"\nEtiquetas:\n"
        f"{LABELS_PATH}"
    )


    print(
        "=" * 75
    )


if __name__ == "__main__":

    main()