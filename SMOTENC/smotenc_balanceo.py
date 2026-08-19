# ==========================================
# TESIS: PREDICCION DE PERDIDA GESTACIONAL
# SMOTENC - BALANCEO DE CLASES (solo sobre el train set)
# EJECUCION LOCAL
# ==========================================
#
# QUE HACE ESTE SCRIPT, EN UNA FRASE:
# toma el dataset top 10 que genero Boruta (boruta/dataset_topN_importancia.csv)
# y balancea las clases del target SOLO en el conjunto de entrenamiento,
# dejando el conjunto de prueba intacto, con la proporcion real 91%/9%.
#
# POR QUE SMOTENC Y NO SMOTE PLANO:
# El SMOTE original genera filas sinteticas interpolando linealmente entre
# vecinos reales de la clase minoritaria. Eso funciona bien si TODAS las
# variables son continuas, pero el top 10 de Boruta mezcla continuas
# (V040, V222, IMC, V012, M14, V201) con categoricas (V190, V025, V106,
# V501). Si se interpolara sobre las categoricas, saldrian valores que no
# existen en la realidad (ej. V025 = 1.4 entre "urbano"=1 y "rural"=2).
# SMOTENC resuelve esto: para las columnas categoricas que se le indican,
# en vez de interpolar toma el valor mas frecuente entre los vecinos usados
# para generar la fila sintetica.
#
# POR QUE SOLO EN EL TRAIN SET:
# Si se balanceara antes de dividir train/test, filas sinteticas del train
# quedarian interpoladas a partir de vecinos que despues caen en el test
# (fuga de datos), inflando artificialmente las metricas de evaluacion. El
# orden correcto es: dividir train/test primero (estratificado, para que
# ambos conserven la proporcion real 91%/9%), balancear DESPUES solo el
# train, y evaluar siempre contra el test original sin balancear.
#
# Dependencias (instalar una sola vez desde la terminal, no aqui adentro):
#   py -m pip install imbalanced-learn pandas scikit-learn

# ==========================================
# PASO 1: IMPORTAR LIBRERIAS
# ==========================================

from pathlib import Path

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC

# ==========================================
# PASO 2: CONFIGURACION DE RUTAS
# ==========================================
# Este script vive en SMOTENC/, un nivel por debajo de la raiz del proyecto.
# Lee el dataset top 10 que ya guardo boruta/boruta_seleccion_caracteristicas.py
# (no vuelve a tocar app.py ni el consolidado, ni el script de Boruta).

BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_ARCHIVO = BASE_DIR / "boruta" / "dataset_topN_importancia.csv"
RUTA_TRAIN_BALANCEADO = BASE_DIR / "SMOTENC" / "train_balanceado.csv"
RUTA_TEST_ORIGINAL = BASE_DIR / "SMOTENC" / "test_original.csv"

# Columnas categoricas del top 10 (ver justificacion arriba). El resto
# (V040, V222, IMC, V012, M14, V201) se trata como continua/conteo.
COLUMNAS_CATEGORICAS = ['V190', 'V025', 'V106', 'V501']

print(f"Buscando archivo en: {RUTA_ARCHIVO}")

# ==========================================
# PASO 3: CARGAR EL DATASET TOP 10 (salida de Boruta)
# ==========================================

print("\n" + "=" * 60)
print("TESIS: PREDICCION DE PERDIDA GESTACIONAL")
print("SMOTENC - BALANCEO DE CLASES")
print("=" * 60)

try:
    df = pd.read_csv(RUTA_ARCHIVO, encoding='utf-8-sig')
    print("Dataset cargado correctamente")
    print(f"Dimensiones: {df.shape[0]:,} filas, {df.shape[1]:,} columnas")
except FileNotFoundError:
    print(f"ERROR: No se encontro el archivo en {RUTA_ARCHIVO}")
    print("Verifica que boruta/boruta_seleccion_caracteristicas.py ya se haya ejecutado")
    raise

print("\nDistribucion de target (dataset completo, antes de dividir):")
print(df['target'].value_counts())
print(f"   Clase 0 (Sin perdida): {(df['target'] == 0).sum():,} ({(df['target'] == 0).mean() * 100:.1f}%)")
print(f"   Clase 1 (Perdida): {(df['target'] == 1).sum():,} ({(df['target'] == 1).mean() * 100:.1f}%)")

# ==========================================
# PASO 4: SEPARAR X (FEATURES) E y (TARGET)
# ==========================================

feature_names = [c for c in df.columns if c != 'target']
X = df[feature_names].copy()
y = df['target'].copy()

print(f"\nFeatures (X): {X.shape}")
print(f"Target (y): {y.shape}")
print(f"Nombres de features ({len(feature_names)}): {feature_names}")

# Indices de las columnas categoricas dentro de X, que es lo que SMOTENC
# necesita (no los nombres, sino la posicion de cada columna).
categorical_features = [feature_names.index(c) for c in COLUMNAS_CATEGORICAS if c in feature_names]
columnas_continuas = [c for c in feature_names if c not in COLUMNAS_CATEGORICAS]

print(f"\nColumnas categoricas ({len(categorical_features)}): {COLUMNAS_CATEGORICAS}")
print(f"Columnas continuas/conteo ({len(columnas_continuas)}): {columnas_continuas}")

# ==========================================
# PASO 5: DIVIDIR EN TRAIN Y TEST (ANTES de balancear)
# ==========================================
# Mismos parametros que en boruta/boruta_seleccion_caracteristicas.py
# (test_size=0.30, random_state=2023) para que el train/test sea
# consistente entre etapas del proyecto. stratify=y mantiene la proporcion
# 91%/9% en ambos conjuntos.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=2023, stratify=y
)

print(f"\nTrain (antes de balancear): {X_train.shape}")
print(f"   Clase 0: {(y_train == 0).sum():,} ({(y_train == 0).mean() * 100:.1f}%)")
print(f"   Clase 1: {(y_train == 1).sum():,} ({(y_train == 1).mean() * 100:.1f}%)")
print(f"\nTest (se queda intacto, proporcion real): {X_test.shape}")
print(f"   Clase 0: {(y_test == 0).sum():,} ({(y_test == 0).mean() * 100:.1f}%)")
print(f"   Clase 1: {(y_test == 1).sum():,} ({(y_test == 1).mean() * 100:.1f}%)")

# ==========================================
# PASO 6: SMOTENC - BALANCEO SOLO DEL TRAIN
# ==========================================
# sampling_strategy='auto' balancea la clase minoritaria hasta igualar a la
# mayoritaria (queda 50%/50%). random_state=2023 por consistencia con el
# resto del proyecto.

print("\n" + "=" * 60)
print("EJECUTANDO SMOTENC SOBRE EL TRAIN SET")
print("=" * 60)

smotenc = SMOTENC(
    categorical_features=categorical_features,
    sampling_strategy='auto',
    random_state=2023,
)

X_train_bal, y_train_bal = smotenc.fit_resample(X_train, y_train)

print(f"\nTrain despues de SMOTENC: {X_train_bal.shape}")
print(f"   Clase 0: {(y_train_bal == 0).sum():,} ({(y_train_bal == 0).mean() * 100:.1f}%)")
print(f"   Clase 1: {(y_train_bal == 1).sum():,} ({(y_train_bal == 1).mean() * 100:.1f}%)")
print(f"   Filas sinteticas generadas: {X_train_bal.shape[0] - X_train.shape[0]:,}")

# ==========================================
# PASO 7: GUARDAR RESULTADOS
# ==========================================
# Se guardan DOS archivos separados, para que quede explicito cual es cual
# en la siguiente etapa (entrenamiento de modelos):
#
#   a) train_balanceado.csv: usar SOLO para entrenar los modelos.
#   b) test_original.csv: usar SOLO para evaluar. Nunca se balancea, porque
#      en produccion el modelo se enfrenta a la proporcion real 91%/9%.

print("\n" + "-" * 60)
print("GUARDANDO RESULTADOS")
print("-" * 60)

df_train_bal = X_train_bal.copy()
df_train_bal['target'] = y_train_bal.values
df_train_bal.to_csv(RUTA_TRAIN_BALANCEADO, index=False, encoding='utf-8-sig')
print(f"Train balanceado guardado en: {RUTA_TRAIN_BALANCEADO}")

df_test = X_test.copy()
df_test['target'] = y_test.values
df_test.to_csv(RUTA_TEST_ORIGINAL, index=False, encoding='utf-8-sig')
print(f"Test original guardado en: {RUTA_TEST_ORIGINAL}")

# ==========================================
# PASO 8: RESUMEN FINAL
# ==========================================

print("\n" + "=" * 60)
print("RESUMEN FINAL - SMOTENC")
print("=" * 60)

print(f"\nDataset original (top 10 de Boruta): {df.shape[0]:,} filas, {df.shape[1]} columnas")
print(f"Train original: {X_train.shape[0]:,} filas (91%/9% real)")
print(f"Train balanceado: {X_train_bal.shape[0]:,} filas (50%/50% sintetico)")
print(f"Test (sin tocar): {X_test.shape[0]:,} filas (91%/9% real)")

print(f"\nArchivos generados en SMOTENC/:")
print(f"   1. train_balanceado.csv (usar para ENTRENAR los modelos)")
print(f"   2. test_original.csv    (usar para EVALUAR los modelos)")

print(f"\nSiguiente paso: segun el modelo a entrenar, puede faltar escalado")
print(f"de las columnas continuas y one-hot encoding de V501 (categorica")
print(f"nominal) para modelos sensibles a distancia/gradiente (SVM, redes")
print(f"neuronales, regresion logistica). Los modelos basados en arboles")
print(f"(Random Forest, XGBoost) pueden usar train_balanceado.csv tal cual.")

print("\nSMOTENC COMPLETADO")
print("=" * 60)
