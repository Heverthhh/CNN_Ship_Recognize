from pathlib import Path
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================================================
# LOCALIZAR AUTOMÁTICAMENTE EL PROYECTO
# =========================================================

# Este archivo está en:
# CNN_Ships/src/probar_thresholds.py

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

nombre_csv = "resultados_clasificador_barcos.csv"

# Buscar el archivo en todo el proyecto
archivos = list(PROJECT_DIR.rglob(nombre_csv))

if not archivos:
    print("\nERROR: No se encontró:")
    print(nombre_csv)

    print("\nProyecto buscado:")
    print(PROJECT_DIR)

    print("\nCopia el CSV dentro de CNN_Ships o alguna subcarpeta.")
    raise SystemExit

CSV = archivos[0]

print("\nCSV ENCONTRADO:")
print(CSV)

# =========================================================
# LEER CSV
# =========================================================

df = pd.read_csv(CSV)

print("\nColumnas encontradas:")
print(df.columns.tolist())

print("\nPrimeras filas:")
print(df.head())

# =========================================================
# DETECTAR NOMBRES DE COLUMNAS
# =========================================================

# Intentar encontrar etiqueta real
posibles_real = [
    "real_label",
    "etiqueta_real",
    "label",
    "true_label",
    "y_true"
]

col_real = None

for col in posibles_real:
    if col in df.columns:
        col_real = col
        break

if col_real is None:
    print("\nERROR: No encuentro la columna de etiqueta real.")
    print("Columnas disponibles:")
    print(df.columns.tolist())
    raise SystemExit


# Intentar encontrar probabilidad de barco
posibles_prob = [
    "prob_ship",
    "prob_barco",
    "probability",
    "probabilidad_barco",
    "score"
]

col_prob = None

for col in posibles_prob:
    if col in df.columns:
        col_prob = col
        break

if col_prob is None:
    print("\nERROR: No encuentro la columna de probabilidad de barco.")
    print("Columnas disponibles:")
    print(df.columns.tolist())
    raise SystemExit


print("\nUsando:")
print("Etiqueta real:", col_real)
print("Probabilidad barco:", col_prob)

# =========================================================
# THRESHOLDS
# =========================================================

thresholds = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    0.97
]

print("\n")
print("=" * 80)
print("RESULTADOS POR THRESHOLD")
print("=" * 80)

resultados = []

for threshold in thresholds:

    y_true = df[col_real].astype(int)

    # Si probabilidad >= threshold => BARCO
    y_pred = (df[col_prob] >= threshold).astype(int)

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

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    resultados.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

    print(f"\nTHRESHOLD = {threshold:.2f}")
    print("-" * 45)

    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1       : {f1 * 100:.2f}%")

    print("\nMatriz de confusión:")
    print(cm)

    print("\nTN =", tn)
    print("FP =", fp)
    print("FN =", fn)
    print("TP =", tp)


# =========================================================
# MEJOR THRESHOLD DE ESTA PRUEBA
# =========================================================

df_resultados = pd.DataFrame(resultados)

mejor = df_resultados.loc[
    df_resultados["accuracy"].idxmax()
]

print("\n")
print("=" * 80)
print("MEJOR RESULTADO EN ESTE CSV")
print("=" * 80)

print(
    f"""
Threshold : {mejor['threshold']:.2f}

Accuracy  : {mejor['accuracy'] * 100:.2f}%
Precision : {mejor['precision'] * 100:.2f}%
Recall    : {mejor['recall'] * 100:.2f}%
F1        : {mejor['f1'] * 100:.2f}%

TN = {int(mejor['TN'])}
FP = {int(mejor['FP'])}
FN = {int(mejor['FN'])}
TP = {int(mejor['TP'])}
"""
)

# =========================================================
# GUARDAR RESULTADOS
# =========================================================

salida = PROJECT_DIR / "resultados_thresholds.csv"

df_resultados.to_csv(
    salida,
    index=False
)

print("Tabla completa guardada en:")
print(salida)
