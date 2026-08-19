# ==========================================
# PRUEBA DE CONTROL: BORUTA SOBRE EL DATASET "PIMA INDIANS DIABETES"
# ==========================================
#
# QUE ES ESTO Y PARA QUE SIRVE:
# diabetes.csv es un dataset publico muy conocido (768 filas, 8 variables
# predictoras + Outcome), sin relacion con la tesis de perdida gestacional.
# Se usa aqui como PRUEBA DE CONTROL: correr exactamente la misma
# metodologia de Boruta (mismos parametros que en boruta/) sobre un dataset
# distinto, mas balanceado (65%/35% en vez de 91%/9%) y sin las
# complicaciones de patrones de salto de encuesta que tiene ENDES.
#
# Si Boruta confirma bastantes mas variables aqui que en el dataset de
# ENDES, es una señal de que el numero bajo de variables confirmadas en la
# tesis (3-4 de ~20) tiene que ver con el desbalance de clases y el ruido
# propio de la encuesta, no con que el script de Boruta este mal armado.
#
# Los parametros de BorutaPy y del RandomForestClassifier son los mismos
# que en boruta/boruta_seleccion_caracteristicas.py, para que la
# comparacion sea justa (mismo metodo, distintos datos).

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from boruta import BorutaPy

BASE_DIR = Path(__file__).resolve().parent
RUTA_ARCHIVO = BASE_DIR / "diabetes.csv"
RUTA_GUARDAR = BASE_DIR / "resultados_boruta_diabetes.csv"

# ==========================================
# CARGAR DATOS
# ==========================================

print("=" * 60)
print("PRUEBA DE CONTROL: BORUTA SOBRE diabetes.csv")
print("=" * 60)

df = pd.read_csv(RUTA_ARCHIVO)
print(f"Dimensiones: {df.shape[0]:,} filas, {df.shape[1]} columnas")
print(f"\nDistribucion de Outcome (target):")
print(df['Outcome'].value_counts())
print(f"   Clase 0: {(df['Outcome'] == 0).sum():,} ({(df['Outcome'] == 0).mean() * 100:.1f}%)")
print(f"   Clase 1: {(df['Outcome'] == 1).sum():,} ({(df['Outcome'] == 1).mean() * 100:.1f}%)")

feature_names = [c for c in df.columns if c != 'Outcome']
X = df[feature_names].values
y = df['Outcome'].values

print(f"\nFeatures ({len(feature_names)}): {feature_names}")

# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=2023, stratify=y
)
print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")

# ==========================================
# BORUTA (mismos parametros que en boruta/boruta_seleccion_caracteristicas.py)
# ==========================================

print("\n" + "=" * 60)
print("EJECUTANDO BORUTA")
print("=" * 60)

rf_boruta_estimator = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    n_jobs=-1,
    random_state=2023,
)

boruta = BorutaPy(
    estimator=rf_boruta_estimator,
    n_estimators='auto',
    max_iter=50,
    alpha=0.05,
    early_stopping=True,
    n_iter_no_change=15,
    random_state=2023,
)

boruta.fit(X_train, y_train)

selected = boruta.support_
ranking = boruta.ranking_

rf_importancias = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    n_jobs=-1,
    random_state=2023,
)
rf_importancias.fit(X_train, y_train)

resultados = pd.DataFrame({
    'Feature': feature_names,
    'Seleccionada': selected,
    'Ranking': ranking,
    'Importancia': rf_importancias.feature_importances_,
})
resultados = resultados.sort_values('Importancia', ascending=False).reset_index(drop=True)
resultados.insert(0, 'Posicion', range(1, len(resultados) + 1))
resultados['Importancia_acumulada_pct'] = (
    resultados['Importancia'].cumsum() / resultados['Importancia'].sum() * 100
).round(1)

print(f"\nResumen de Boruta:")
print(f"   Total de features evaluadas: {len(feature_names)}")
print(f"   Features confirmadas: {selected.sum()}")
print(f"   Features rechazadas: {(~selected).sum()}")

print(f"\nTabla completa (de mayor a menor importancia):")
print(resultados.to_string(index=False))

resultados.to_csv(RUTA_GUARDAR, index=False, encoding='utf-8-sig')
print(f"\nResultados guardados en: {RUTA_GUARDAR}")

# ==========================================
# GRAFICO
# ==========================================

plt.figure(figsize=(8, 5))
orden_grafico = resultados.iloc[::-1]
colores = ['green' if s else 'red' for s in orden_grafico['Seleccionada']]
plt.barh(orden_grafico['Feature'], orden_grafico['Importancia'], color=colores)
plt.xlabel('Importancia (Random Forest)')
plt.title('Boruta sobre diabetes.csv\n(Verde = Confirmada, Rojo = Rechazada)')
plt.tight_layout()
plt.savefig(BASE_DIR / "grafico_importancia_diabetes.png", dpi=150)
print(f"Grafico guardado en: {BASE_DIR / 'grafico_importancia_diabetes.png'}")

# ==========================================
# VALIDACION
# ==========================================

print("\n" + "-" * 60)
print("VALIDACION: TODAS LAS FEATURES VS. SOLO LAS CONFIRMADAS")
print("-" * 60)


def evaluar(nombre, y_true, y_pred):
    print(f"\n{nombre}")
    print(f"   Accuracy:            {accuracy_score(y_true, y_pred):.4f}")
    print(f"   Balanced accuracy:   {balanced_accuracy_score(y_true, y_pred):.4f}")
    print(f"   Precision (clase 1): {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"   Recall (clase 1):    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"   F1 (clase 1):        {f1_score(y_true, y_pred, zero_division=0):.4f}")


rf_all = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=2023)
rf_all.fit(X_train, y_train)
evaluar(f"Todas las features ({X_train.shape[1]})", y_test, rf_all.predict(X_test))

if selected.sum() > 0:
    rf_sel = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=2023)
    rf_sel.fit(X_train[:, selected], y_train)
    evaluar(f"Solo confirmadas por Boruta ({selected.sum()})", y_test, rf_sel.predict(X_test[:, selected]))

print("\n" + "=" * 60)
print("PRUEBA DE CONTROL COMPLETADA")
print("=" * 60)
