# ==========================================
# TESIS: PREDICCION DE PERDIDA GESTACIONAL
# ENTRENAMIENTO Y COMPARACION DE MODELOS
# EJECUCION LOCAL
# ==========================================
#
# QUE HACE ESTE SCRIPT, EN UNA FRASE:
# entrena los 5 algoritmos de la Tabla 5 (Random Forest, Gradient Boosting,
# XGBoost, KNN, SVM) sobre el train balanceado por SMOTENC, los evalua
# contra el test original (sin balancear), y arma una tabla comparativa.
#
# POR QUE DOS GRUPOS DE MODELOS CON DISTINTO PREPROCESAMIENTO:
# Random Forest, Gradient Boosting y XGBoost dividen el espacio por
# umbrales (ej. "V012 > 30?"), asi que no les afecta la escala de las
# variables ni que una categorica este codificada como entero. Usan
# train_balanceado.csv tal cual.
#
# KNN y SVM comparan DISTANCIAS entre puntos. Si no se escalan las
# variables continuas, la que tenga los numeros mas grandes (V040,
# altitud, que llega a miles) domina el calculo de distancia solo por su
# magnitud, no porque sea mas importante. Y si V501 (estado civil, sin
# orden real) se deja como 1,2,3,4,5, el modelo interpreta que "separada"
# esta lejos de "casada" solo porque 5 esta lejos de 1 en la recta
# numerica, cuando son etiquetas sin jerarquia. Por eso a KNN y SVM se les
# aplica antes: StandardScaler (empareja la escala de las variables
# continuas/ordinales) + OneHotEncoder (convierte V501 en columnas
# binarias independientes, sin orden implicito).
#
# Dependencias (instalar una sola vez desde la terminal, no aqui adentro):
#   py -m pip install pandas scikit-learn matplotlib xgboost

# ==========================================
# PASO 1: IMPORTAR LIBRERIAS
# ==========================================

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ==========================================
# PASO 2: CONFIGURACION DE RUTAS
# ==========================================
# Este script vive en Modelos/, un nivel por debajo de la raiz del
# proyecto. Lee el train balanceado y el test original que ya genero
# SMOTENC/smotenc_balanceo.py (no se vuelve a tocar ese script).

BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_TRAIN = BASE_DIR / "SMOTENC" / "train_balanceado.csv"
RUTA_TEST = BASE_DIR / "SMOTENC" / "test_original.csv"
RUTA_GUARDAR = BASE_DIR / "Modelos" / "resultados_modelos.csv"
RUTA_GRAFICO = BASE_DIR / "Modelos" / "grafico_comparacion_modelos.png"

# V501 (estado civil) es la unica variable NOMINAL del top 10: sus
# categorias no tienen un orden real, asi que necesita one-hot encoding
# para KNN/SVM. V190 (quintil de riqueza) y V106 (nivel educativo) SI son
# ordinales (tienen un orden real), y V025 (urbano/rural) es binaria, asi
# que las tres se tratan como numericas y solo se escalan, sin one-hot.
COLUMNA_NOMINAL = ['V501']

RANDOM_STATE = 2023

print(f"Buscando archivos en: {RUTA_TRAIN.parent}")

# ==========================================
# PASO 3: CARGAR TRAIN BALANCEADO Y TEST ORIGINAL
# ==========================================

print("\n" + "=" * 60)
print("TESIS: PREDICCION DE PERDIDA GESTACIONAL")
print("ENTRENAMIENTO Y COMPARACION DE MODELOS")
print("=" * 60)

try:
    df_train = pd.read_csv(RUTA_TRAIN, encoding='utf-8-sig')
    df_test = pd.read_csv(RUTA_TEST, encoding='utf-8-sig')
    print("Datasets cargados correctamente")
except FileNotFoundError:
    print("ERROR: no se encontraron train_balanceado.csv / test_original.csv")
    print("Verifica que SMOTENC/smotenc_balanceo.py ya se haya ejecutado")
    raise

print(f"\nTrain (balanceado por SMOTENC): {df_train.shape[0]:,} filas, {df_train.shape[1]} columnas")
print(f"   Clase 0: {(df_train['target'] == 0).sum():,} ({(df_train['target'] == 0).mean() * 100:.1f}%)")
print(f"   Clase 1: {(df_train['target'] == 1).sum():,} ({(df_train['target'] == 1).mean() * 100:.1f}%)")

print(f"\nTest (proporcion real, sin balancear): {df_test.shape[0]:,} filas, {df_test.shape[1]} columnas")
print(f"   Clase 0: {(df_test['target'] == 0).sum():,} ({(df_test['target'] == 0).mean() * 100:.1f}%)")
print(f"   Clase 1: {(df_test['target'] == 1).sum():,} ({(df_test['target'] == 1).mean() * 100:.1f}%)")

# ==========================================
# PASO 4: SEPARAR X (FEATURES) E y (TARGET)
# ==========================================

feature_names = [c for c in df_train.columns if c != 'target']
columnas_escalar = [c for c in feature_names if c not in COLUMNA_NOMINAL]

X_train = df_train[feature_names]
y_train = df_train['target'].astype(int)
X_test = df_test[feature_names]
y_test = df_test['target'].astype(int)

print(f"\nFeatures ({len(feature_names)}): {feature_names}")
print(f"Columna nominal (one-hot para KNN/SVM): {COLUMNA_NOMINAL}")
print(f"Columnas a escalar (KNN/SVM): {columnas_escalar}")

# ==========================================
# PASO 5: PREPROCESAMIENTO PARA KNN Y SVM
# ==========================================
# ColumnTransformer aplica una transformacion distinta por grupo de
# columnas: StandardScaler a las continuas/ordinales/binaria, y
# OneHotEncoder solo a V501. Se mete dentro de un Pipeline junto con el
# modelo para que el ajuste (fit) del escalador y del encoder se haga
# SOLO con el train (evita fuga de datos hacia el test).

preprocesador = ColumnTransformer(transformers=[
    ('escalado', StandardScaler(), columnas_escalar),
    ('onehot', OneHotEncoder(handle_unknown='ignore'), COLUMNA_NOMINAL),
])

# ==========================================
# PASO 6: DEFINIR LOS 5 MODELOS
# ==========================================
# Random Forest, Gradient Boosting y XGBoost entrenan directamente sobre
# X_train/X_test (sin pipeline de preprocesamiento). KNN y SVM se envuelven
# en un Pipeline que primero preprocesa y luego entrena.
#
# No se agrega class_weight='balanced' en ningun modelo: el train ya viene
# balanceado 50/50 por SMOTENC, asi que ponderar de nuevo por clase seria
# redundante (las clases ya pesan igual en cantidad de filas).

modelos = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1,
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, max_depth=3, random_state=RANDOM_STATE,
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200, max_depth=6, eval_metric='logloss',
        random_state=RANDOM_STATE, n_jobs=-1,
    ),
    'KNN': Pipeline(steps=[
        ('preprocesar', preprocesador),
        ('modelo', KNeighborsClassifier(n_neighbors=5)),
    ]),
    'SVM': Pipeline(steps=[
        ('preprocesar', preprocesador),
        ('modelo', SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)),
    ]),
}

# ==========================================
# PASO 7: ENTRENAR Y EVALUAR CADA MODELO
# ==========================================
# Mismas metricas que en boruta/boruta_seleccion_caracteristicas.py, mas
# AUC-ROC (para poder comparar contra las cifras de la literatura citada en
# la tesis, que reportan AUC ademas de exactitud). Con un target
# desbalanceado en el test (~91/9), accuracy sola es enganosa: un modelo
# que siempre prediga "0" ya tendria ~91% de accuracy sin aprender nada.

print("\n" + "=" * 60)
print("ENTRENANDO Y EVALUANDO MODELOS")
print("=" * 60)

resultados = []

for nombre, modelo in modelos.items():
    print(f"\nEntrenando {nombre}...")
    modelo.fit(X_train, y_train)

    pred = modelo.predict(X_test)
    proba = modelo.predict_proba(X_test)[:, 1]

    metricas = {
        'Modelo': nombre,
        'Accuracy': accuracy_score(y_test, pred),
        'Balanced_Accuracy': balanced_accuracy_score(y_test, pred),
        'Precision_clase1': precision_score(y_test, pred, zero_division=0),
        'Recall_clase1': recall_score(y_test, pred, zero_division=0),
        'F1_clase1': f1_score(y_test, pred, zero_division=0),
        'AUC_ROC': roc_auc_score(y_test, proba),
    }
    resultados.append(metricas)

    print(f"   Accuracy:            {metricas['Accuracy']:.4f}")
    print(f"   Balanced accuracy:   {metricas['Balanced_Accuracy']:.4f}")
    print(f"   Precision (clase 1): {metricas['Precision_clase1']:.4f}")
    print(f"   Recall (clase 1):    {metricas['Recall_clase1']:.4f}")
    print(f"   F1 (clase 1):        {metricas['F1_clase1']:.4f}")
    print(f"   AUC-ROC:             {metricas['AUC_ROC']:.4f}")

# ==========================================
# PASO 8: TABLA COMPARATIVA FINAL
# ==========================================
# Se ordena por F1 de la clase 1 (equilibrio entre precision y recall para
# detectar casos de perdida gestacional), no por accuracy, que con este
# nivel de desbalance en el test no distingue bien un modelo util de uno
# que solo predice la clase mayoritaria.

df_resultados = pd.DataFrame(resultados)
df_resultados = df_resultados.sort_values('F1_clase1', ascending=False).reset_index(drop=True)
df_resultados.insert(0, 'Posicion', range(1, len(df_resultados) + 1))

print("\n" + "-" * 60)
print("TABLA COMPARATIVA (ordenada por F1, clase 1)")
print("-" * 60)
print(df_resultados.to_string(index=False))

# ==========================================
# PASO 9: GRAFICO COMPARATIVO
# ==========================================

plt.figure(figsize=(9, 5))
orden_grafico = df_resultados.iloc[::-1]
plt.barh(orden_grafico['Modelo'], orden_grafico['F1_clase1'], color='steelblue')
plt.xlabel('F1-score (clase 1: perdida gestacional)')
plt.title('Comparacion de modelos sobre test set\n(train balanceado con SMOTENC)')
plt.tight_layout()
plt.savefig(RUTA_GRAFICO, dpi=150)
print(f"\nGrafico guardado en: {RUTA_GRAFICO}")

# ==========================================
# PASO 10: GUARDAR RESULTADOS Y RESUMEN FINAL
# ==========================================

df_resultados.to_csv(RUTA_GUARDAR, index=False, encoding='utf-8-sig')
print(f"Tabla de resultados guardada en: {RUTA_GUARDAR}")

print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
print(f"\nMejor modelo por F1 (clase 1): {df_resultados.iloc[0]['Modelo']}")
print(f"   F1: {df_resultados.iloc[0]['F1_clase1']:.4f} | AUC-ROC: {df_resultados.iloc[0]['AUC_ROC']:.4f}")

print(f"\nArchivos generados en Modelos/:")
print(f"   1. resultados_modelos.csv           (tabla comparativa de los 5 modelos)")
print(f"   2. grafico_comparacion_modelos.png  (grafico de barras por F1)")

print("\nENTRENAMIENTO COMPLETADO")
print("=" * 60)
