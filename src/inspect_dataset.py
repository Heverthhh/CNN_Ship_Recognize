from pathlib import Path
from collections import Counter, defaultdict

from PIL import Image
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset" / "raw"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FUNCIÓN PARA INTERPRETAR EL NOMBRE
# ============================================================

def parse_filename(filename):
    """
    Formato esperado aproximado:
    label__scene_id__longitude_latitude.png

    Ejemplo:
    1__20180712_180321_0f2a__-122.35_37.81.png
    """

    stem = Path(filename).stem

    try:
        parts = stem.split("__")

        label = int(parts[0])

        scene_id = parts[1] if len(parts) > 1 else "UNKNOWN"

        coordinates = parts[2] if len(parts) > 2 else ""

        coord_parts = coordinates.split("_")

        longitude = (
            float(coord_parts[0])
            if len(coord_parts) >= 1 and coord_parts[0]
            else None
        )

        latitude = (
            float(coord_parts[1])
            if len(coord_parts) >= 2 and coord_parts[1]
            else None
        )

        return {
            "label": label,
            "scene_id": scene_id,
            "longitude": longitude,
            "latitude": latitude,
        }

    except Exception:
        return {
            "label": None,
            "scene_id": "PARSE_ERROR",
            "longitude": None,
            "latitude": None,
        }


# ============================================================
# INSPECCIÓN DEL DATASET
# ============================================================

def inspect_dataset():

    print("=" * 70)
    print("SHIPSNET DATASET INSPECTOR")
    print("=" * 70)

    if not DATASET_DIR.exists():
        print(f"\nERROR: No existe la carpeta:\n{DATASET_DIR}")
        return

    files = sorted(DATASET_DIR.glob("*.png"))

    print(f"\nImágenes encontradas: {len(files)}")

    if len(files) == 0:
        print("\nNo se encontraron imágenes PNG.")
        return

    records = []

    corrupted = []

    for i, file in enumerate(files):

        metadata = parse_filename(file.name)

        try:

            with Image.open(file) as img:

                width, height = img.size
                mode = img.mode

                record = {
                    "filename": file.name,
                    "label": metadata["label"],
                    "class_name": (
                        "ship"
                        if metadata["label"] == 1
                        else "no_ship"
                        if metadata["label"] == 0
                        else "unknown"
                    ),
                    "scene_id": metadata["scene_id"],
                    "longitude": metadata["longitude"],
                    "latitude": metadata["latitude"],
                    "width": width,
                    "height": height,
                    "mode": mode,
                }

                records.append(record)

        except Exception as e:

            corrupted.append({
                "filename": file.name,
                "error": str(e)
            })

        if (i + 1) % 500 == 0:
            print(f"Procesadas: {i + 1}/{len(files)}")

    df = pd.DataFrame(records)

    # ========================================================
    # RESUMEN GENERAL
    # ========================================================

    print("\n")
    print("=" * 70)
    print("RESUMEN")
    print("=" * 70)

    print(f"\nImágenes válidas: {len(df)}")
    print(f"Imágenes corruptas: {len(corrupted)}")

    print("\nDistribución de clases:")

    if not df.empty:

        print(
            df["class_name"]
            .value_counts()
            .to_string()
        )

        print("\nResoluciones encontradas:")

        resolutions = (
            df.groupby(["width", "height"])
            .size()
            .sort_values(ascending=False)
        )

        print(resolutions.to_string())

        print("\nModos de color:")

        print(
            df["mode"]
            .value_counts()
            .to_string()
        )

        print("\nNúmero de scene_id únicos:")

        print(df["scene_id"].nunique())

        print("\nNúmero de imágenes por escena:")

        scenes = (
            df["scene_id"]
            .value_counts()
            .sort_values(ascending=False)
        )

        print(scenes.to_string())

        # ====================================================
        # VERIFICACIONES
        # ====================================================

        print("\n")
        print("=" * 70)
        print("VERIFICACIONES")
        print("=" * 70)

        invalid_size = df[
            (df["width"] != 80) |
            (df["height"] != 80)
        ]

        print(
            f"\nImágenes que NO son 80x80: "
            f"{len(invalid_size)}"
        )

        invalid_rgb = df[
            df["mode"] != "RGB"
        ]

        print(
            f"Imágenes que NO son RGB: "
            f"{len(invalid_rgb)}"
        )

        invalid_labels = df[
            ~df["label"].isin([0, 1])
        ]

        print(
            f"Imágenes con etiqueta inválida: "
            f"{len(invalid_labels)}"
        )

        # ====================================================
        # BALANCE DE CLASES
        # ====================================================

        print("\n")
        print("=" * 70)
        print("BALANCE DE CLASES")
        print("=" * 70)

        counts = df["label"].value_counts()

        total = len(df)

        for label, count in counts.items():

            name = "SHIP" if label == 1 else "NO SHIP"

            percentage = 100 * count / total

            print(
                f"{name:10s}: "
                f"{count:5d} "
                f"({percentage:.2f}%)"
            )

        # ====================================================
        # INFORMACIÓN POR ESCENA
        # ====================================================

        scene_summary = (
            df.groupby("scene_id")
            .agg(
                total_images=("filename", "count"),
                ships=("label", lambda x: (x == 1).sum()),
                no_ships=("label", lambda x: (x == 0).sum()),
            )
            .reset_index()
        )

        # ====================================================
        # EXPORTACIÓN
        # ====================================================

        dataset_csv = RESULTS_DIR / "dataset_metadata.csv"

        scene_csv = RESULTS_DIR / "scene_summary.csv"

        df.to_csv(
            dataset_csv,
            index=False
        )

        scene_summary.to_csv(
            scene_csv,
            index=False
        )

        print("\n")
        print("=" * 70)
        print("ARCHIVOS GENERADOS")
        print("=" * 70)

        print(f"\n{dataset_csv}")
        print(f"{scene_csv}")

    if corrupted:

        corrupted_df = pd.DataFrame(corrupted)

        corrupted_csv = (
            RESULTS_DIR /
            "corrupted_images.csv"
        )

        corrupted_df.to_csv(
            corrupted_csv,
            index=False
        )

        print(
            f"\nReporte de imágenes corruptas:"
            f"\n{corrupted_csv}"
        )

    print("\n")
    print("=" * 70)
    print("AUDITORÍA FINALIZADA")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    inspect_dataset()