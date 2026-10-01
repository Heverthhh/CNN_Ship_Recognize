from pathlib import Path
import zipfile
import random
import shutil
import math

import cv2
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ZIP_CANDIDATES = [
    BASE_DIR / "ships_aerial_images.zip",
    BASE_DIR / "archive(1).zip",
]

OUTPUT_DIR = (
    BASE_DIR
    / "test_externo_balanceado_v3"
)

IMAGES_DIR = (
    OUTPUT_DIR
    / "images"
)

LABELS_CSV = (
    OUTPUT_DIR
    / "labels.csv"
)

REPORT_CSV = (
    OUTPUT_DIR
    / "generation_report.csv"
)

PREVIEW_PATH = (
    OUTPUT_DIR
    / "preview_v3.jpg"
)


# ============================================================
# CONFIGURACIÓN DEL TEST
# ============================================================

FINAL_SIZE = 80

N_SHIP = 200
N_NO_SHIP = 200

RANDOM_SEED = 42


# ============================================================
# FILTROS DE CALIDAD PARA BARCOS
# ============================================================

# Ningún lado del bbox puede ser menor a 8 px.
MIN_BBOX_DIMENSION = 8.0

# Evitar cajas extremadamente delgadas/deformadas.
# Ej:
# 100 x 5  → ratio 20 → rechazada
MAX_ASPECT_RATIO = 12.0

# Queremos que el lado mayor del barco ocupe
# aproximadamente 60% del chip.
TARGET_OCCUPANCY = 0.60

# Permitimos este rango después de crear el crop.
MIN_OCCUPANCY = 0.50
MAX_OCCUPANCY = 0.70

# Tamaño mínimo del crop original.
MIN_SOURCE_CROP = 24

# No ponemos MAX_CROP_SIZE arbitrario.
# El límite real será el tamaño de la imagen.


# ============================================================
# NEGATIVOS
# ============================================================

MAX_NEGATIVE_ATTEMPTS = 150

# Separación mínima adicional respecto a barcos.
NEGATIVE_MARGIN_RATIO = 0.10


# ============================================================
# BUSCAR ZIP
# ============================================================

def find_zip():

    for path in ZIP_CANDIDATES:

        if path.exists():
            return path

    raise FileNotFoundError(
        "\nNo encontré el ZIP externo.\n\n"
        "Coloca uno de estos archivos en la raíz:\n"
        "ships_aerial_images.zip\n"
        "archive(1).zip\n"
    )


# ============================================================
# POSIBLE SHIPSNET
# ============================================================

def is_shipsnet_filename(filename):

    name = Path(filename).name

    return (
        name.startswith("0__")
        or
        name.startswith("1__")
    )


# ============================================================
# DECODIFICAR IMAGEN
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
# PARSEAR LABEL YOLO
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

            xc_norm = float(
                parts[1]
            )

            yc_norm = float(
                parts[2]
            )

            w_norm = float(
                parts[3]
            )

            h_norm = float(
                parts[4]
            )

        except ValueError:
            continue

        # Una sola clase:
        # 0 = ship
        if class_id != 0:
            continue

        # Rechazar datos YOLO inválidos
        if not (
            0 <= xc_norm <= 1
            and
            0 <= yc_norm <= 1
            and
            0 < w_norm <= 1
            and
            0 < h_norm <= 1
        ):
            continue

        cx = (
            xc_norm
            *
            width
        )

        cy = (
            yc_norm
            *
            height
        )

        box_w = (
            w_norm
            *
            width
        )

        box_h = (
            h_norm
            *
            height
        )

        x1 = (
            cx
            -
            box_w / 2
        )

        y1 = (
            cy
            -
            box_h / 2
        )

        x2 = (
            cx
            +
            box_w / 2
        )

        y2 = (
            cy
            +
            box_h / 2
        )

        # Limitar bbox a imagen
        x1 = max(
            0.0,
            x1
        )

        y1 = max(
            0.0,
            y1
        )

        x2 = min(
            float(width),
            x2
        )

        y2 = min(
            float(height),
            y2
        )

        real_w = (
            x2 - x1
        )

        real_h = (
            y2 - y1
        )

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
                y2
        })

    return boxes


# ============================================================
# VALIDAR BOUNDING BOX
# ============================================================

def validate_ship_box(
    box,
    image_width,
    image_height
):

    box_w = box[
        "width"
    ]

    box_h = box[
        "height"
    ]

    # --------------------------------------------------------
    # Dimensiones mínimas
    # --------------------------------------------------------

    if (
        box_w < MIN_BBOX_DIMENSION
        or
        box_h < MIN_BBOX_DIMENSION
    ):

        return (
            False,
            "bbox_demasiado_pequeno"
        )

    # --------------------------------------------------------
    # Aspect ratio
    # --------------------------------------------------------

    long_side = max(
        box_w,
        box_h
    )

    short_side = min(
        box_w,
        box_h
    )

    aspect_ratio = (
        long_side
        /
        short_side
    )

    if (
        aspect_ratio
        >
        MAX_ASPECT_RATIO
    ):

        return (
            False,
            "aspect_ratio_extremo"
        )

    # --------------------------------------------------------
    # Tamaño crop necesario
    # --------------------------------------------------------

    required_crop = (
        long_side
        /
        TARGET_OCCUPANCY
    )

    # Si el crop necesario no cabe en la imagen,
    # descartamos el barco.
    if (
        required_crop
        >
        min(
            image_width,
            image_height
        )
    ):

        return (
            False,
            "barco_demasiado_grande"
        )

    return (
        True,
        "ok"
    )


# ============================================================
# CALCULAR CROP CUADRADO
# ============================================================

def calculate_square_crop(
    box,
    image_width,
    image_height
):

    long_side = max(
        box["width"],
        box["height"]
    )

    # --------------------------------------------------------
    # Tamaño del crop para que el barco ocupe ~60%
    # --------------------------------------------------------

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

    # El crop DEBE caber completo en imagen
    if (
        crop_size > image_width
        or
        crop_size > image_height
    ):

        return None

    cx = box[
        "cx"
    ]

    cy = box[
        "cy"
    ]

    # Posición ideal
    x0 = int(
        round(
            cx
            -
            crop_size / 2
        )
    )

    y0 = int(
        round(
            cy
            -
            crop_size / 2
        )
    )

    # --------------------------------------------------------
    # Ajustar bordes
    # --------------------------------------------------------

    x0 = max(
        0,
        min(
            x0,
            image_width - crop_size
        )
    )

    y0 = max(
        0,
        min(
            y0,
            image_height - crop_size
        )
    )

    x1 = (
        x0
        +
        crop_size
    )

    y1 = (
        y0
        +
        crop_size
    )

    # --------------------------------------------------------
    # GARANTIZAR que bbox quede COMPLETO dentro
    # --------------------------------------------------------

    if box["x1"] < x0:
        return None

    if box["y1"] < y0:
        return None

    if box["x2"] > x1:
        return None

    if box["y2"] > y1:
        return None

    # --------------------------------------------------------
    # Ocupación resultante
    # --------------------------------------------------------

    occupancy = (
        max(
            box["width"],
            box["height"]
        )
        /
        crop_size
    )

    if not (
        MIN_OCCUPANCY
        <=
        occupancy
        <=
        MAX_OCCUPANCY
    ):

        return None

    return {

        "x0":
            x0,

        "y0":
            y0,

        "x1":
            x1,

        "y1":
            y1,

        "crop_size":
            crop_size,

        "occupancy":
            occupancy
    }


# ============================================================
# GENERAR CHIP BARCO
# ============================================================

def generate_ship_crop(
    image,
    box
):

    image_h, image_w = (
        image.shape[:2]
    )

    valid, reason = (
        validate_ship_box(
            box,
            image_w,
            image_h
        )
    )

    if not valid:

        return (
            None,
            None,
            reason
        )

    crop_info = (
        calculate_square_crop(
            box,
            image_w,
            image_h
        )
    )

    if crop_info is None:

        return (
            None,
            None,
            "crop_invalido"
        )

    x0 = crop_info[
        "x0"
    ]

    y0 = crop_info[
        "y0"
    ]

    x1 = crop_info[
        "x1"
    ]

    y1 = crop_info[
        "y1"
    ]

    crop = image[
        y0:y1,
        x0:x1
    ]

    if crop.size == 0:

        return (
            None,
            None,
            "crop_vacio"
        )

    # --------------------------------------------------------
    # Redimensionar a 80x80
    # --------------------------------------------------------

    crop_80 = cv2.resize(
        crop,
        (
            FINAL_SIZE,
            FINAL_SIZE
        ),
        interpolation=cv2.INTER_AREA
    )

    return (
        crop_80,
        crop_info,
        "ok"
    )


# ============================================================
# INTERSECCIÓN
# ============================================================

def rectangles_intersect(
    rect,
    box,
    margin=0
):

    x1, y1, x2, y2 = (
        rect
    )

    bx1 = (
        box["x1"]
        -
        margin
    )

    by1 = (
        box["y1"]
        -
        margin
    )

    bx2 = (
        box["x2"]
        +
        margin
    )

    by2 = (
        box["y2"]
        +
        margin
    )

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
# GENERAR NEGATIVO
# ============================================================

def generate_negative_crop(
    image,
    boxes,
    source_crop_size,
    rng
):

    image_h, image_w = (
        image.shape[:2]
    )

    crop_size = int(
        source_crop_size
    )

    if (
        crop_size > image_w
        or
        crop_size > image_h
    ):

        return (
            None,
            None
        )

    margin = int(
        round(
            crop_size
            *
            NEGATIVE_MARGIN_RATIO
        )
    )

    margin = max(
        margin,
        4
    )

    for _ in range(
        MAX_NEGATIVE_ATTEMPTS
    ):

        x0 = rng.randint(
            0,
            image_w - crop_size
        )

        y0 = rng.randint(
            0,
            image_h - crop_size
        )

        x1 = (
            x0
            +
            crop_size
        )

        y1 = (
            y0
            +
            crop_size
        )

        rect = (
            x0,
            y0,
            x1,
            y1
        )

        # ----------------------------------------------------
        # No puede tocar ningún barco
        # ----------------------------------------------------

        collision = any(

            rectangles_intersect(
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

        crop_80 = cv2.resize(
            crop,
            (
                FINAL_SIZE,
                FINAL_SIZE
            ),
            interpolation=cv2.INTER_AREA
        )

        return (
            crop_80,
            {
                "x0":
                    x0,

                "y0":
                    y0,

                "x1":
                    x1,

                "y1":
                    y1,

                "crop_size":
                    crop_size
            }
        )

    return (
        None,
        None
    )


# ============================================================
# CREAR PREVIEW
# ============================================================

def create_preview(
    records
):

    if not records:
        return

    positives = [
        r
        for r in records
        if r["label"] == 1
    ][:20]

    negatives = [
        r
        for r in records
        if r["label"] == 0
    ][:20]

    selected = (
        positives
        +
        negatives
    )

    tiles = []

    for record in selected:

        path = (
            IMAGES_DIR
            /
            record[
                "filename"
            ]
        )

        image = cv2.imread(
            str(path)
        )

        if image is None:
            continue

        image = cv2.resize(
            image,
            (
                120,
                120
            ),
            interpolation=cv2.INTER_NEAREST
        )

        label = (
            "SHIP"
            if record[
                "label"
            ] == 1
            else "NO SHIP"
        )

        cv2.putText(
            image,
            label,
            (
                5,
                18
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (
                255,
                255,
                255
            ),
            1,
            cv2.LINE_AA
        )

        tiles.append(
            image
        )

    if not tiles:
        return

    columns = 8

    rows = math.ceil(
        len(tiles)
        /
        columns
    )

    blank = np.zeros(
        (
            120,
            120,
            3
        ),
        dtype=np.uint8
    )

    while (
        len(tiles)
        <
        rows * columns
    ):

        tiles.append(
            blank.copy()
        )

    row_images = []

    for row in range(
        rows
    ):

        row_tiles = (
            tiles[
                row * columns:
                (row + 1) * columns
            ]
        )

        row_images.append(
            np.hstack(
                row_tiles
            )
        )

    montage = np.vstack(
        row_images
    )

    cv2.imwrite(
        str(
            PREVIEW_PATH
        ),
        montage
    )


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
        "GENERADOR DE TEST EXTERNO BALANCEADO V3"
    )

    print(
        "=" * 75
    )

    print(
        f"\nZIP:\n{zip_path}"
    )


    # ========================================================
    # LIMPIAR CARPETA DE SALIDA
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
    # LEER ZIP
    # ========================================================

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as zf:

        names = (
            zf.namelist()
        )

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

        print(
            f"\nImágenes test: "
            f"{len(test_images)}"
        )

        # ----------------------------------------------------
        # Filtrar posible ShipsNet
        # ----------------------------------------------------

        test_images = [

            name

            for name in test_images

            if not is_shipsnet_filename(
                name
            )
        ]

        print(
            f"Después de filtrar ShipsNet: "
            f"{len(test_images)}"
        )

        rng.shuffle(
            test_images
        )


        # ====================================================
        # CARGAR DATASET
        # ====================================================

        dataset = []

        print(
            "\nLeyendo imágenes y anotaciones..."
        )

        for index, image_member in enumerate(
            test_images
        ):

            try:

                image_data = (
                    zf.read(
                        image_member
                    )
                )

                image = decode_image(
                    image_data
                )

            except Exception:
                continue

            if image is None:
                continue

            height, width = (
                image.shape[:2]
            )

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
                width,
                height
            )

            dataset.append({

                "source":
                    image_member,

                "image":
                    image,

                "boxes":
                    boxes,

                "width":
                    width,

                "height":
                    height
            })

            if (
                (index + 1)
                %
                300
                ==
                0
            ):

                print(
                    f"Leídas: "
                    f"{index + 1}"
                    f"/"
                    f"{len(test_images)}"
                )

        print(
            f"\nImágenes válidas: "
            f"{len(dataset)}"
        )


        # ====================================================
        # CANDIDATOS POSITIVOS
        # ====================================================

        rejection_stats = {}

        valid_candidates = []

        for item_index, item in enumerate(
            dataset
        ):

            for box_index, box in enumerate(
                item[
                    "boxes"
                ]
            ):

                (
                    crop,
                    crop_info,
                    reason
                ) = generate_ship_crop(
                    item[
                        "image"
                    ],
                    box
                )

                if crop is None:

                    rejection_stats[
                        reason
                    ] = (
                        rejection_stats.get(
                            reason,
                            0
                        )
                        +
                        1
                    )

                    continue

                valid_candidates.append({

                    "item_index":
                        item_index,

                    "box_index":
                        box_index,

                    "crop":
                        crop,

                    "crop_info":
                        crop_info,

                    "box":
                        box
                })


        print(
            f"\nCandidatos BARCO válidos: "
            f"{len(valid_candidates)}"
        )

        print(
            "\nCajas rechazadas:"
        )

        if rejection_stats:

            for reason, count in (
                rejection_stats.items()
            ):

                print(
                    f"  {reason}: "
                    f"{count}"
                )

        else:

            print(
                "  Ninguna"
            )


        # ====================================================
        # POSITIVOS:
        # PRIMERO UNA CAJA POR IMAGEN FUENTE
        # ====================================================

        grouped = {}

        for candidate in (
            valid_candidates
        ):

            grouped.setdefault(
                candidate[
                    "item_index"
                ],
                []
            ).append(
                candidate
            )

        item_indexes = list(
            grouped.keys()
        )

        rng.shuffle(
            item_indexes
        )

        selected_positive = []

        used_candidate_ids = set()


        # ----------------------------------------------------
        # PASADA 1:
        # una embarcación por imagen fuente
        # ----------------------------------------------------

        for item_index in (
            item_indexes
        ):

            if (
                len(
                    selected_positive
                )
                >=
                N_SHIP
            ):

                break

            candidates = grouped[
                item_index
            ].copy()

            rng.shuffle(
                candidates
            )

            candidate = (
                candidates[0]
            )

            selected_positive.append(
                candidate
            )

            used_candidate_ids.add(
                (
                    candidate[
                        "item_index"
                    ],
                    candidate[
                        "box_index"
                    ]
                )
            )


        # ----------------------------------------------------
        # PASADA 2:
        # otras cajas si todavía faltan
        # ----------------------------------------------------

        if (
            len(
                selected_positive
            )
            <
            N_SHIP
        ):

            remaining = [

                candidate

                for candidate
                in valid_candidates

                if (
                    candidate[
                        "item_index"
                    ],
                    candidate[
                        "box_index"
                    ]
                )
                not in
                used_candidate_ids
            ]

            rng.shuffle(
                remaining
            )

            for candidate in (
                remaining
            ):

                if (
                    len(
                        selected_positive
                    )
                    >=
                    N_SHIP
                ):

                    break

                selected_positive.append(
                    candidate
                )


        # ====================================================
        # GUARDAR POSITIVOS
        # ====================================================

        records = []

        positive_crop_sizes = []

        print(
            "\nGuardando BARCO..."
        )

        for index, candidate in enumerate(
            selected_positive,
            start=1
        ):

            item = dataset[
                candidate[
                    "item_index"
                ]
            ]

            box = candidate[
                "box"
            ]

            crop_info = candidate[
                "crop_info"
            ]

            filename = (
                f"external_ship_"
                f"{index:04d}.png"
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

            positive_crop_sizes.append(
                crop_info[
                    "crop_size"
                ]
            )

            long_side = max(
                box[
                    "width"
                ],
                box[
                    "height"
                ]
            )

            short_side = min(
                box[
                    "width"
                ],
                box[
                    "height"
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

                "bbox_width":
                    round(
                        box[
                            "width"
                        ],
                        3
                    ),

                "bbox_height":
                    round(
                        box[
                            "height"
                        ],
                        3
                    ),

                "aspect_ratio":
                    round(
                        long_side
                        /
                        short_side,
                        3
                    ),

                "source_crop_size":
                    crop_info[
                        "crop_size"
                    ],

                "ship_occupancy":
                    round(
                        crop_info[
                            "occupancy"
                        ],
                        4
                    ),

                "crop_x0":
                    crop_info[
                        "x0"
                    ],

                "crop_y0":
                    crop_info[
                        "y0"
                    ],

                "crop_x1":
                    crop_info[
                        "x1"
                    ],

                "crop_y1":
                    crop_info[
                        "y1"
                    ]
            })


        ship_count = len(
            selected_positive
        )

        print(
            f"BARCO generados: "
            f"{ship_count}"
        )


        # ====================================================
        # NEGATIVOS
        # ====================================================

        print(
            "\nGenerando NO BARCO..."
        )

        negative_count = 0

        used_negative = set()

        attempts = 0

        while (
            negative_count
            <
            N_NO_SHIP
            and
            attempts
            <
            20000
        ):

            attempts += 1

            item_index = (
                rng.randrange(
                    len(
                        dataset
                    )
                )
            )

            item = dataset[
                item_index
            ]

            if positive_crop_sizes:

                source_crop_size = (
                    rng.choice(
                        positive_crop_sizes
                    )
                )

            else:

                source_crop_size = (
                    FINAL_SIZE
                )

            (
                crop,
                crop_info
            ) = generate_negative_crop(
                item[
                    "image"
                ],
                item[
                    "boxes"
                ],
                source_crop_size,
                rng
            )

            if crop is None:
                continue

            key = (
                item[
                    "source"
                ],
                crop_info[
                    "x0"
                ],
                crop_info[
                    "y0"
                ],
                crop_info[
                    "crop_size"
                ]
            )

            if key in used_negative:
                continue

            used_negative.add(
                key
            )

            negative_count += 1

            filename = (
                f"external_noship_"
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

                "bbox_width":
                    None,

                "bbox_height":
                    None,

                "aspect_ratio":
                    None,

                "source_crop_size":
                    crop_info[
                        "crop_size"
                    ],

                "ship_occupancy":
                    0.0,

                "crop_x0":
                    crop_info[
                        "x0"
                    ],

                "crop_y0":
                    crop_info[
                        "y0"
                    ],

                "crop_x1":
                    crop_info[
                        "x1"
                    ],

                "crop_y1":
                    crop_info[
                        "y1"
                    ]
            })


        print(
            f"NO BARCO generados: "
            f"{negative_count}"
        )


    # ========================================================
    # DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        records
    )

    if not df.empty:

        df = df.sample(
            frac=1,
            random_state=RANDOM_SEED
        ).reset_index(
            drop=True
        )

    df.to_csv(
        LABELS_CSV,
        index=False
    )

    df.to_csv(
        REPORT_CSV,
        index=False
    )


    # ========================================================
    # PREVIEW
    # ========================================================

    create_preview(
        records
    )


    # ========================================================
    # ESTADÍSTICAS
    # ========================================================

    positive_df = df[
        df[
            "label"
        ] == 1
    ]

    print("\n")
    print(
        "=" * 75
    )

    print(
        "RESULTADOS V3"
    )

    print(
        "=" * 75
    )

    print(
        f"\nBARCO: "
        f"{ship_count}"
    )

    print(
        f"NO BARCO: "
        f"{negative_count}"
    )

    print(
        f"TOTAL: "
        f"{len(df)}"
    )


    if not positive_df.empty:

        print(
            "\n--- CALIDAD DE POSITIVOS ---"
        )

        print(
            "BBox ancho mínimo: "
            f"{positive_df['bbox_width'].min():.2f}px"
        )

        print(
            "BBox alto mínimo: "
            f"{positive_df['bbox_height'].min():.2f}px"
        )

        print(
            "Aspect ratio máximo: "
            f"{positive_df['aspect_ratio'].max():.2f}"
        )

        print(
            "Ocupación media del barco: "
            f"{positive_df['ship_occupancy'].mean() * 100:.2f}%"
        )

        print(
            "Ocupación mínima: "
            f"{positive_df['ship_occupancy'].min() * 100:.2f}%"
        )

        print(
            "Ocupación máxima: "
            f"{positive_df['ship_occupancy'].max() * 100:.2f}%"
        )

        print(
            "Fuentes distintas usadas: "
            f"{positive_df['source_image'].nunique()}"
        )


    print(
        f"\nImágenes:\n"
        f"{IMAGES_DIR}"
    )

    print(
        f"\nCSV etiquetas:\n"
        f"{LABELS_CSV}"
    )

    print(
        f"\nPreview visual:\n"
        f"{PREVIEW_PATH}"
    )


    # ========================================================
    # VERIFICACIÓN FINAL
    # ========================================================

    if (
        ship_count == N_SHIP
        and
        negative_count == N_NO_SHIP
    ):

        print(
            "\n✓ TEST EXTERNO V3 CREADO CORRECTAMENTE"
        )

    else:

        print(
            "\n⚠ NO SE CONSIGUIERON LAS 400 IMÁGENES"
        )

        print(
            "Esto puede ser correcto si los filtros "
            "de calidad eliminaron demasiados barcos."
        )

    print(
        "=" * 75
    )


if __name__ == "__main__":

    main()