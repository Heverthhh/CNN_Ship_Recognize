from pathlib import Path
import cv2

from features import (
    extract_hog,
    extract_hsv,
    extract_lbp,
    extract_features
)


BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    BASE_DIR
    / "dataset"
    / "raw"
)


files = list(DATASET_DIR.glob("*.png"))

if not files:
    raise RuntimeError(
        "No se encontraron imágenes."
    )


image_path = files[0]

image = cv2.imread(
    str(image_path)
)


hog_features = extract_hog(image)
hsv_features = extract_hsv(image)
lbp_features = extract_lbp(image)

all_features = extract_features(image)


print("=" * 50)
print("PRUEBA DE EXTRACCIÓN")
print("=" * 50)

print(f"\nImagen:")
print(image_path.name)

print("\nCaracterísticas:")

print(
    f"HOG: {len(hog_features)}"
)

print(
    f"HSV:  {len(hsv_features)}"
)

print(
    f"LBP:  {len(lbp_features)}"
)

print(
    f"\nTOTAL: {len(all_features)}"
)

print("\nExtracción correcta ✓")