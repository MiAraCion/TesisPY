# ==========================================
# TESIS: PREDICCION DE PERDIDA GESTACIONAL
# FASE 2: VALIDACION DEL EFECTO DE SMOTE-NC (CON vs. SIN)
# EJECUCION LOCAL
# ==========================================
#
# QUE HACE ESTE SCRIPT, EN UNA FRASE:
# entrena el MISMO modelo (Random Forest) sobre el MISMO train/test split,
# una vez SIN balancear las clases y otra vez CON SMOTENC, para medir el
# efecto real del balanceo de forma aislada (HE2 de la matriz de
# consistencia: "SMOTE-NC incrementa significativamente el Recall y el
# F1-Score frente al entrenamiento con clases desbalanceadas").
#
# POR QUE UN MODELO FIJO Y NO LOS 5:
# el objetivo de este script no es comparar algoritmos (eso ya se hace en
# Modelos/), es aislar el efecto de UNA sola variable metodologica: el
# balanceo. Si se compararan 5 modelos x 2 condiciones, el efecto de
# SMOTENC se mezclaria con las diferencias entre algoritmos. Se usa Random
# Forest porque fue el modelo mas consistente en los resultados previos del
# proyecto.
#
# METRICAS: ademas de Accuracy/Precision/Recall/F1/AUC-ROC (ya usadas en
# Modelos/), se agrega PR-AUC (average_precision_score) porque la matriz de
# consistencia lo pide explicitamente para esta fase: con un desbalance tan
# fuerte (~91/9), el AUC-ROC puede verse "bien" aunque el modelo apenas
# distinga la clase minoritaria, mientras que PR-AUC es mas exigente porque
# no se ve favorecido por la enorme cantidad de verdaderos negativos.
#
# Dependencias (instalar una sola vez desde la terminal, no aqui adentro):
#   py -m pip install pandas scikit-learn matplotlib imbalanced-learn

# ==========================================
# PASO 1: IMPORTAR LIBRERIAS
# ==========================================

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    precision_recall_curve,
)
from imblearn.over_sampling import SMOTENC

RANDOM_STATE = 2023

# ==========================================
# PASO 2: CONFIGURACION DE RUTAS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_ARCHIVO = BASE_DIR / "boruta" / "dataset_topN_importancia.csv"
RUTA_GUARDAR = BASE_DIR / "Validacion_SMOTE" / "comparacion_con_sin_smote.csv"
RUTA_GRAFICO = BASE_DIR / "Validacion_SMOTE" / "grafico_con_sin_smote.png"
RUTA_GRAFICO_PR = BASE_DIR / "Validacion_SMOTE" / "curva_precision_recall.png"

COLUMNAS_CATEGORICAS = ['V190', 'V025', 'V106', 'V501', 'QS206']

print(f"Buscando archivo en: {RUTA_ARCHIVO}")

print("\n" + "=" * 60)
print("TESIS: PREDICCION DE PERDIDA GESTACIONAL")
print("FASE 2: VALIDACION SMOTE-NC (CON vs. SIN)")
print("=" * 60)

# ==========================================
# PASO 3: CARGAR DATOS Y DIVIDIR TRAIN/TEST (identico a SMOTENC/smotenc_balanceo.py)
# ==========================================

df = pd.read_csv(RUTA_ARCHIVO, encoding='utf-8-sig')
feature_names = [c for c in df.columns if c != 'target']
X = df[feature_names].copy()
y = df['target'].copy()

categorical_features = [feature_names.index(c) for c in COLUMNAS_CATEGORICAS if c in feature_names]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTrain (sin tocar): {X_train.shape[0]:,} filas "
      f"({(y_train == 0).sum():,} clase 0 / {(y_train == 1).sum():,} clase 1)")
print(f"Test (mismo para ambas condiciones): {X_test.shape[0]:,} filas "
      f"({(y_test == 0).sum():,} clase 0 / {(y_test == 1).sum():,} clase 1)")

# ==========================================
# PASO 4: CONDICION A - SIN BALANCEAR (datos "sin procesar")
# ==========================================
# Random Forest por defecto, SIN class_weight y SIN SMOTENC: el modelo ve
# el train tal cual viene (~91%/9%), que es exactamente la comparacion base
# que pide HE2 ("frente al entrenamiento con clases desbalanceadas").

print("\n" + "-" * 60)
print("CONDICION A: SIN SMOTE-NC (train desbalanceado, ~91%/9%)")
print("-" * 60)

modelo_sin_smote = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
modelo_sin_smote.fit(X_train, y_train)

pred_sin = modelo_sin_smote.predict(X_test)
proba_sin = modelo_sin_smote.predict_proba(X_test)[:, 1]

# ==========================================
# PASO 5: CONDICION B - CON SMOTENC (train balanceado 50/50)
# ==========================================

print("\n" + "-" * 60)
print("CONDICION B: CON SMOTE-NC (train balanceado 50/50)")
print("-" * 60)

smotenc = SMOTENC(categorical_features=categorical_features, sampling_strategy='auto', random_state=RANDOM_STATE)
X_train_bal, y_train_bal = smotenc.fit_resample(X_train, y_train)
print(f"Train balanceado: {X_train_bal.shape[0]:,} filas "
      f"({(y_train_bal == 0).sum():,} clase 0 / {(y_train_bal == 1).sum():,} clase 1)")
print(f"Tasa de sobremuestreo sintetico: 1:1 "
      f"(se generaron {X_train_bal.shape[0] - X_train.shape[0]:,} filas sinteticas de clase 1)")

modelo_con_smote = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
modelo_con_smote.fit(X_train_bal, y_train_bal)

pred_con = modelo_con_smote.predict(X_test)
proba_con = modelo_con_smote.predict_proba(X_test)[:, 1]

# ==========================================
# PASO 6: COMPARAR METRICAS (mismo test set en ambos casos)
# ==========================================

print("\n" + "=" * 60)
print("COMPARACION DE METRICAS (mismo test, sin tocar en ambos casos)")
print("=" * 60)


def calcular_metricas(nombre, y_true, y_pred, y_proba):
    metricas = {
        'Condicion': nombre,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced_Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Precision_clase1': precision_score(y_true, y_pred, zero_division=0),
        'Recall_clase1': recall_score(y_true, y_pred, zero_division=0),
        'F1_clase1': f1_score(y_true, y_pred, zero_division=0),
        'AUC_ROC': roc_auc_score(y_true, y_proba),
        'PR_AUC': average_precision_score(y_true, y_proba),
    }
    print(f"\n{nombre}")
    for k, v in metricas.items():
        if k != 'Condicion':
            print(f"   {k}: {v:.4f}")
    return metricas


fila_sin = calcular_metricas("SIN SMOTE-NC", y_test, pred_sin, proba_sin)
fila_con = calcular_metricas("CON SMOTE-NC", y_test, pred_con, proba_con)

# Delta: fila CON menos fila SIN, columna por columna numerica
delta = {}
for col in ['Accuracy', 'Balanced_Accuracy', 'Precision_clase1', 'Recall_clase1', 'F1_clase1', 'AUC_ROC', 'PR_AUC']:
    delta[col] = fila_con[col] - fila_sin[col]

print("\n" + "-" * 60)
print("EFECTO DE SMOTE-NC (CON - SIN)")
print("-" * 60)
for k, v in delta.items():
    signo = "+" if v >= 0 else ""
    print(f"   Delta {k}: {signo}{v:.4f}")

evidencia_he2 = delta['Recall_clase1'] > 0 and delta['F1_clase1'] > 0
print(f"\nHE2 ('SMOTE-NC incrementa Recall y F1 frente a datos desbalanceados'): "
      f"{'CONFIRMADA' if evidencia_he2 else 'NO CONFIRMADA'} con este modelo y este dataset")

# ==========================================
# PASO 7: GUARDAR TABLA Y GRAFICOS
# ==========================================

df_final = pd.DataFrame([fila_sin, fila_con])
df_final.to_csv(RUTA_GUARDAR, index=False, encoding='utf-8-sig')
print(f"\nTabla guardada en: {RUTA_GUARDAR}")

# Grafico de barras: Recall y F1 con vs sin
plt.figure(figsize=(7, 5))
metricas_grafico = ['Recall_clase1', 'F1_clase1', 'PR_AUC', 'AUC_ROC']
x = range(len(metricas_grafico))
plt.bar([i - 0.2 for i in x], [fila_sin[m] for m in metricas_grafico], width=0.4, label='Sin SMOTE-NC', color='lightcoral')
plt.bar([i + 0.2 for i in x], [fila_con[m] for m in metricas_grafico], width=0.4, label='Con SMOTE-NC', color='seagreen')
plt.xticks(list(x), metricas_grafico, rotation=15)
plt.ylabel('Valor de la metrica')
plt.title('Efecto de SMOTE-NC sobre el mismo modelo y mismo test set')
plt.legend()
plt.tight_layout()
plt.savefig(RUTA_GRAFICO, dpi=150)
print(f"Grafico guardado en: {RUTA_GRAFICO}")

# Curva Precision-Recall (indicador explicito de la Fase 2 en la matriz de consistencia)
precision_sin, recall_sin, _ = precision_recall_curve(y_test, proba_sin)
precision_con, recall_con, _ = precision_recall_curve(y_test, proba_con)

plt.figure(figsize=(7, 5))
plt.plot(recall_sin, precision_sin, label=f"Sin SMOTE-NC (PR-AUC={fila_sin['PR_AUC']:.3f})", color='lightcoral')
plt.plot(recall_con, precision_con, label=f"Con SMOTE-NC (PR-AUC={fila_con['PR_AUC']:.3f})", color='seagreen')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall: efecto de SMOTE-NC')
plt.legend()
plt.tight_layout()
plt.savefig(RUTA_GRAFICO_PR, dpi=150)
print(f"Curva Precision-Recall guardada en: {RUTA_GRAFICO_PR}")

print("\n" + "=" * 60)
print("VALIDACION SMOTE-NC COMPLETADA")
print("=" * 60)
