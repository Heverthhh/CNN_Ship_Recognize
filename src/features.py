import cv2
import numpy as np

from skimage.feature import hog, local_binary_pattern


# ============================================================
# CONFIGURACIÓN
# ============================================================

IMAGE_SIZE = (80, 80)

LBP_POINTS = 8
LBP_RADIUS = 1


# ============================================================
# PREPROCESAMIENTO
# ============================================================

def preprocess_image(image):
    """
    Garantiza:
    - Imagen válida
    - 3 canales BGR
    - Tamaño 80x80
    """

    if image is None:
        raise ValueError("La imagen está vacía.")

    # Escala de grises -> BGR
    if len(image.shape) == 2:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

    # BGRA -> BGR
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2BGR
        )

    # Garantizar 80x80
    if image.shape[:2] != IMAGE_SIZE:
        image = cv2.resize(
            image,
            IMAGE_SIZE,
            interpolation=cv2.INTER_AREA
        )

    return image


# ============================================================
# HOG
# ============================================================

def extract_hog(image):

    image = preprocess_image(image)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )

    return features.astype(
        np.float32
    )


# ============================================================
# HSV
# ============================================================

def extract_hsv(image):

    image = preprocess_image(image)

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    feature_vector = []

    for channel in range(3):

        hist = cv2.calcHist(
            [hsv],
            [channel],
            None,
            [16],
            [0, 256]
        )

        hist = cv2.normalize(
            hist,
            hist
        ).flatten()

        feature_vector.extend(
            hist
        )

    return np.array(
        feature_vector,
        dtype=np.float32
    )


# ============================================================
# LBP
# ============================================================

def extract_lbp(image):

    image = preprocess_image(image)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    lbp = local_binary_pattern(
        gray,
        P=LBP_POINTS,
        R=LBP_RADIUS,
        method="uniform"
    )

    bins = LBP_POINTS + 2

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(
            0,
            bins + 1
        ),
        range=(0, bins)
    )

    hist = hist.astype(
        np.float32
    )

    hist /= (
        hist.sum() + 1e-7
    )

    return hist


# ============================================================
# HOG + HSV
# MODELO FINAL
# ============================================================

def extract_hog_hsv(image):

    image = preprocess_image(
        image
    )

    hog_features = extract_hog(
        image
    )

    hsv_features = extract_hsv(
        image
    )

    features = np.concatenate([
        hog_features,
        hsv_features
    ])

    return features.astype(
        np.float32
    )


# ============================================================
# HOG + HSV + LBP
# ============================================================

def extract_features(image):

    image = preprocess_image(
        image
    )

    hog_features = extract_hog(
        image
    )

    hsv_features = extract_hsv(
        image
    )

    lbp_features = extract_lbp(
        image
    )

    combined = np.concatenate([
        hog_features,
        hsv_features,
        lbp_features
    ])

    return combined.astype(
        np.float32
    )


# ============================================================
# EXTRAER DESDE ARCHIVO
# ============================================================

def extract_features_from_file(
    path,
    use_lbp=False
):

    image = cv2.imread(
        str(path)
    )

    if image is None:
        raise ValueError(
            f"No se pudo abrir la imagen:\n{path}"
        )

    if use_lbp:

        return extract_features(
            image
        )

    return extract_hog_hsv(
        image
    )