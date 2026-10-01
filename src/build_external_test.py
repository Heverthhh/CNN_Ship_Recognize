from pathlib import Path
import zipfile
import hashlib
import shutil
import tempfile


# ============================================================
# RUTAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset utilizado para entrenamiento
TRAIN_DIR = BASE_DIR / "dataset" / "raw"

# ZIP nuevo que conseguiste
EXTERNAL_ZIP = BASE_DIR / "shipsnet.zip"

# Salidas
OUTPUT_CLEAN = BASE_DIR / "test_externo_sin_repetidos"
OUTPUT_NEW_SCENES = BASE_DIR / "test_externo_scene_nueva"


# ============================================================
# UTILIDADES
# ============================================================

def get_scene_id(filename):
    """
    Formato esperado:
    label__scene_id__longitude_latitude.png
    """

    stem = Path(filename).stem
    parts = stem.split("__")

    if len(parts) >= 2:
        return parts[1]

    return "UNKNOWN"


def get_label(filename):

    try:
        return int(
            Path(filename).name.split("__")[0]
        )

    except Exception:
        return None


def file_hash(path):
    """
    SHA256 para detectar imágenes idénticas aunque
    eventualmente tengan nombres diferentes.
    """

    sha = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            block = f.read(1024 * 1024)

            if not block:
                break

            sha.update(block)

    return sha.hexdigest()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GENERADOR DE TEST EXTERNO LIMPIO")
    print("=" * 70)

    # --------------------------------------------------------
    # Verificaciones
    # --------------------------------------------------------

    if not TRAIN_DIR.exists():

        raise FileNotFoundError(
            f"No existe el dataset de entrenamiento:\n{TRAIN_DIR}"
        )

    if not EXTERNAL_ZIP.exists():

        raise FileNotFoundError(
            f"No existe el ZIP externo:\n{EXTERNAL_ZIP}"
        )

    # --------------------------------------------------------
    # Crear / limpiar carpetas de salida
    # --------------------------------------------------------

    for folder in [
        OUTPUT_CLEAN,
        OUTPUT_NEW_SCENES
    ]:

        if folder.exists():
            shutil.rmtree(folder)

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

    # ========================================================
    # DATASET DE ENTRENAMIENTO
    # ========================================================

    train_files = sorted(
        TRAIN_DIR.glob("*.png")
    )

    print(
        f"\nImágenes de entrenamiento: "
        f"{len(train_files)}"
    )

    train_names = {
        path.name
        for path in train_files
    }

    train_scenes = {
        get_scene_id(path.name)
        for path in train_files
    }

    print(
        f"Scene IDs de entrenamiento: "
        f"{len(train_scenes)}"
    )

    # --------------------------------------------------------
    # HASHES DE TRAIN
    # --------------------------------------------------------

    print(
        "\nCalculando hashes del entrenamiento..."
    )

    train_hashes = set()

    for i, path in enumerate(train_files):

        train_hashes.add(
            file_hash(path)
        )

        if (i + 1) % 500 == 0:

            print(
                f"Hashes: "
                f"{i + 1}/{len(train_files)}"
            )

    # ========================================================
    # EXTRAER ZIP TEMPORALMENTE
    # ========================================================

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_dir = Path(temp_dir)

        print(
            "\nExtrayendo ZIP externo..."
        )

        with zipfile.ZipFile(
            EXTERNAL_ZIP,
            "r"
        ) as zip_ref:

            zip_ref.extractall(
                temp_dir
            )

        external_files = sorted(
            temp_dir.rglob("*.png")
        )

        print(
            f"Imágenes encontradas en ZIP: "
            f"{len(external_files)}"
        )

        # ====================================================
        # CONTADORES
        # ====================================================

        duplicate_name = 0
        duplicate_hash = 0

        clean_count = 0
        new_scene_count = 0

        clean_ship = 0
        clean_no_ship = 0

        new_scene_ship = 0
        new_scene_no_ship = 0

        new_scenes = set()

        # ====================================================
        # ANALIZAR IMÁGENES
        # ====================================================

        for i, path in enumerate(
            external_files
        ):

            filename = path.name

            label = get_label(
                filename
            )

            scene_id = get_scene_id(
                filename
            )

            # ------------------------------------------------
            # DUPLICADO POR NOMBRE
            # ------------------------------------------------

            if filename in train_names:

                duplicate_name += 1

                continue

            # ------------------------------------------------
            # DUPLICADO POR CONTENIDO
            # ------------------------------------------------

            image_hash = file_hash(
                path
            )

            if image_hash in train_hashes:

                duplicate_hash += 1

                continue

            # ------------------------------------------------
            # IMAGEN LIMPIA
            # ------------------------------------------------

            destination = (
                OUTPUT_CLEAN
                / filename
            )

            shutil.copy2(
                path,
                destination
            )

            clean_count += 1

            if label == 1:
                clean_ship += 1

            elif label == 0:
                clean_no_ship += 1

            # ------------------------------------------------
            # SCENE ID COMPLETAMENTE NUEVO
            # ------------------------------------------------

            if scene_id not in train_scenes:

                destination_new = (
                    OUTPUT_NEW_SCENES
                    / filename
                )

                shutil.copy2(
                    path,
                    destination_new
                )

                new_scene_count += 1

                new_scenes.add(
                    scene_id
                )

                if label == 1:
                    new_scene_ship += 1

                elif label == 0:
                    new_scene_no_ship += 1

            if (i + 1) % 500 == 0:

                print(
                    f"Analizadas: "
                    f"{i + 1}/{len(external_files)}"
                )

    # ========================================================
    # RESULTADOS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("RESULTADOS")
    print("=" * 70)

    print(
        f"\nImágenes del ZIP: "
        f"{len(external_files)}"
    )

    print(
        f"Duplicadas por nombre: "
        f"{duplicate_name}"
    )

    print(
        f"Duplicadas por contenido: "
        f"{duplicate_hash}"
    )

    print("\n--- TEST SIN REPETIDOS ---")

    print(
        f"Total: {clean_count}"
    )

    print(
        f"Barco: {clean_ship}"
    )

    print(
        f"No barco: {clean_no_ship}"
    )

    print(
        f"\nCarpeta:\n{OUTPUT_CLEAN}"
    )

    print("\n--- TEST CON SCENES NUEVAS ---")

    print(
        f"Total: {new_scene_count}"
    )

    print(
        f"Barco: {new_scene_ship}"
    )

    print(
        f"No barco: {new_scene_no_ship}"
    )

    print(
        f"Scene IDs completamente nuevos: "
        f"{len(new_scenes)}"
    )

    print(
        f"\nCarpeta:\n{OUTPUT_NEW_SCENES}"
    )

    print("\n")
    print("=" * 70)

    if new_scene_count > 0:

        print(
            "RECOMENDACIÓN: usa test_externo_scene_nueva "
            "como prueba más estricta."
        )

    else:

        print(
            "No hay escenas completamente nuevas. "
            "Usa test_externo_sin_repetidos como prueba adicional."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()