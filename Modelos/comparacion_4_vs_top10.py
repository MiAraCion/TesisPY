# ==========================================
# TESIS: PREDICCION DE PERDIDA GESTACIONAL
# C1: COMPARACION 6 VARIABLES BORUTA-CONFIRMADAS vs. TOP-10 HIBRIDO
# EJECUCION LOCAL
# ==========================================
#
# QUE HACE ESTE SCRIPT, EN UNA FRASE:
# repite la busqueda de hiperparametros + evaluacion de
# Modelos/entrenamiento_modelos_optimizado.py, pero usando SOLO las 6
# variables que Boruta confirmo con evidencia puramente estadistica tras el
# fix C3 (V201, V222, V040, V012, IMC, V190 - ver boruta/resultados_boruta.csv),
# para comparar directamente contra los resultados ya existentes del
# criterio hibrido de 10 variables (HE1).
#
# NOTA: la corrida anterior de este script uso por error una lista vieja de
# "4 confirmadas" (V040, V222, IMC, V190), que correspondia al Boruta
# ANTES del fix C3. Tras C3, Boruta confirma 6 variables, no 4 (se suman
# V012 y V201). Esta version usa la lista correcta.
#
# Dependencias (instalar una sola vez desde la terminal, no aqui adentro):
#   py -m pip install pandas scikit-learn matplotlib xgboost imbalanced-learn

from pathlib import Path

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve,
)
from imblearn.over_sampling import SMOTENC

RANDOM_STATE = 2023
BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_ARCHIVO = BASE_DIR / "boruta" / "dataset_topN_importancia.csv"  # ya incluye las 4 confirmadas
RUTA_TOP10_RESULTADOS = BASE_DIR / "Modelos" / "resultados_modelos_optimizado.csv"
RUTA_GUARDAR = BASE_DIR / "Modelos" / "resultados_6confirmadas.csv"
RUTA_COMPARACION = BASE_DIR / "Modelos" / "comparacion_4_vs_top10.csv"

VARIABLES_CONFIRMADAS = ['V040', 'V222', 'IMC', 'V190', 'V012', 'V201']  # las 6 confirmadas por Boruta tras C3
COLUMNAS_CATEGORICAS = ['V190']  # unica categorica entre las 6 (V012, V201, V222, V040, IMC son continuas/conteo)

print("=" * 60)
print("TESIS: PREDICCION DE PERDIDA GESTACIONAL")
print("C1: 4 VARIABLES BORUTA-CONFIRMADAS vs. TOP-10 HIBRIDO")
print("=" * 60)

df = pd.read_csv(RUTA_ARCHIVO, encoding='utf-8-sig')
feature_names = VARIABLES_CONFIRMADAS
X = df[feature_names].copy()
y = df['target'].astype(int).copy()
print(f"\nDataset: {df.shape[0]:,} filas | Features usadas (solo las 4 confirmadas): {feature_names}")

categorical_features = [feature_names.index(c) for c in COLUMNAS_CATEGORICAS if c in feature_names]
columnas_escalar = [c for c in feature_names if c not in COLUMNAS_CATEGORICAS]

# Mismo split en 3 partes que entrenamiento_modelos_optimizado.py (65/15/20)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.1875, random_state=RANDOM_STATE, stratify=y_train_full
)
print(f"Train: {X_train.shape[0]:,} | Validacion: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}")

smotenc = SMOTENC(categorical_features=categorical_features, sampling_strategy='auto', random_state=RANDOM_STATE)
X_train_bal, y_train_bal = smotenc.fit_resample(X_train, y_train)
print(f"Train balanceado: {X_train_bal.shape[0]:,} filas")

preprocesador = ColumnTransformer(transformers=[('escalado', StandardScaler(), columnas_escalar)], remainder='passthrough')

espacios = {
    'Random Forest': {'n_estimators': [100, 200, 300, 400], 'max_depth': [5, 8, 12, None], 'min_samples_leaf': [1, 3, 5]},
    'Gradient Boosting': {'n_estimators': [100, 150, 200], 'max_depth': [2, 3, 4], 'learning_rate': [0.03, 0.05, 0.1]},
    'XGBoost': {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7], 'learning_rate': [0.03, 0.05, 0.1, 0.2]},
    'KNN': {'n_neighbors': [3, 5, 7, 9, 11, 15], 'weights': ['uniform', 'distance']},
    'SVM': {'C': [0.5, 1, 3, 5], 'gamma': ['scale', 'auto']},
}
n_iter_por_modelo = {'Random Forest': 6, 'Gradient Boosting': 5, 'XGBoost': 6, 'KNN': 6, 'SVM': 4}


def construir_modelo(nombre, params):
    if nombre == 'Random Forest':
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)
    if nombre == 'Gradient Boosting':
        return GradientBoostingClassifier(random_state=RANDOM_STATE, **params)
    if nombre == 'XGBoost':
        return XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1, **params)
    if nombre == 'KNN':
        return Pipeline(steps=[('preprocesar', preprocesador), ('modelo', KNeighborsClassifier(**params))])
    if nombre == 'SVM':
        return Pipeline(steps=[('preprocesar', preprocesador), ('modelo', SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE, **params))])
    raise ValueError(nombre)


def mejor_umbral(y_true, proba):
    precisiones, recalls, umbrales = precision_recall_curve(y_true, proba)
    f1s = np.where((precisiones + recalls) > 0, 2 * precisiones * recalls / (precisiones + recalls + 1e-12), 0)
    mejor_idx = np.argmax(f1s[:-1]) if len(umbrales) > 0 else 0
    return umbrales[mejor_idx] if len(umbrales) > 0 else 0.5


print("\n" + "=" * 60)
print("BUSQUEDA DE HIPERPARAMETROS (evaluada en el set de VALIDACION)")
print("=" * 60)

resultados = []
for nombre, espacio in espacios.items():
    print(f"\n--- {nombre} ---")
    candidatos = list(ParameterSampler(espacio, n_iter=n_iter_por_modelo[nombre], random_state=RANDOM_STATE))

    mejor_f1_val, mejor_params = -1, None
    for params in candidatos:
        modelo = construir_modelo(nombre, params)
        modelo.fit(X_train_bal, y_train_bal)
        pred_val = modelo.predict(X_val)
        f1_val = f1_score(y_val, pred_val, zero_division=0)
        if f1_val > mejor_f1_val:
            mejor_f1_val, mejor_params = f1_val, params
    print(f"   Mejores hiperparametros: {mejor_params} (F1 validacion: {mejor_f1_val:.4f})")

    modelo_final = construir_modelo(nombre, mejor_params)
    modelo_final.fit(X_train_bal, y_train_bal)
    proba_val = modelo_final.predict_proba(X_val)[:, 1]
    umbral = mejor_umbral(y_val, proba_val)

    proba_test = modelo_final.predict_proba(X_test)[:, 1]
    pred_umbral = (proba_test >= umbral).astype(int)
    auc = roc_auc_score(y_test, proba_test)

    fila = {
        'Modelo': nombre,
        'Mejores_hiperparametros': str(mejor_params),
        'Umbral_optimo': round(umbral, 4),
        'AUC_ROC': auc,
        'F1_umbral_optimo': f1_score(y_test, pred_umbral, zero_division=0),
        'Recall_umbral_optimo': recall_score(y_test, pred_umbral, zero_division=0),
        'Precision_umbral_optimo': precision_score(y_test, pred_umbral, zero_division=0),
        'Accuracy_umbral_optimo': accuracy_score(y_test, pred_umbral),
    }
    resultados.append(fila)
    print(f"   TEST -> AUC={auc:.4f} F1={fila['F1_umbral_optimo']:.4f} Recall={fila['Recall_umbral_optimo']:.4f} Precision={fila['Precision_umbral_optimo']:.4f}")

df_resultados = pd.DataFrame(resultados).sort_values('F1_umbral_optimo', ascending=False).reset_index(drop=True)
df_resultados.to_csv(RUTA_GUARDAR, index=False, encoding='utf-8-sig')
print(f"\nTabla (solo 6 confirmadas) guardada en: {RUTA_GUARDAR}")

# ==========================================
# COMPARACION DIRECTA CONTRA EL TOP-10 YA EXISTENTE
# ==========================================
print("\n" + "=" * 60)
print("COMPARACION: 6 CONFIRMADAS vs. TOP-10 HIBRIDO (mismo test, umbral optimo)")
print("=" * 60)

try:
    df_top10 = pd.read_csv(RUTA_TOP10_RESULTADOS, encoding='utf-8-sig')
    comparacion = []
    for _, fila4 in df_resultados.iterrows():
        fila10 = df_top10[df_top10['Modelo'] == fila4['Modelo']]
        if len(fila10) == 0:
            continue
        fila10 = fila10.iloc[0]
        comparacion.append({
            'Modelo': fila4['Modelo'],
            'AUC_6confirmadas': fila4['AUC_ROC'],
            'AUC_Top10': fila10['AUC_ROC'],
            'Delta_AUC': fila4['AUC_ROC'] - fila10['AUC_ROC'],
            'F1_6confirmadas': fila4['F1_umbral_optimo'],
            'F1_Top10': fila10['F1_umbral_optimo'],
            'Delta_F1': fila4['F1_umbral_optimo'] - fila10['F1_umbral_optimo'],
        })
        print(f"\n{fila4['Modelo']}")
        print(f"   AUC:  6-confirmadas={fila4['AUC_ROC']:.4f}  vs  Top-10={fila10['AUC_ROC']:.4f}  (delta={fila4['AUC_ROC']-fila10['AUC_ROC']:+.4f})")
        print(f"   F1:   6-confirmadas={fila4['F1_umbral_optimo']:.4f}  vs  Top-10={fila10['F1_umbral_optimo']:.4f}  (delta={fila4['F1_umbral_optimo']-fila10['F1_umbral_optimo']:+.4f})")

    df_comp = pd.DataFrame(comparacion)
    df_comp.to_csv(RUTA_COMPARACION, index=False, encoding='utf-8-sig')
    print(f"\nTabla comparativa guardada en: {RUTA_COMPARACION}")

    mejora_neta = (df_comp['Delta_F1'] < 0).sum()  # cuantos modelos empeoran con solo 4 variables
    print(f"\nHE1 ('el criterio hibrido de 10 variables mantiene capacidad predictiva igual o superior a las 4 confirmadas'):")
    print(f"   {mejora_neta} de {len(df_comp)} modelos tienen MEJOR F1 con el Top-10 que con las 6 confirmadas")
    veredicto = "CONFIRMADA" if mejora_neta >= len(df_comp) / 2 else "NO CONFIRMADA (las 4 confirmadas igualan o superan al Top-10 en la mayoria de modelos)"
    print(f"   Veredicto: {veredicto}")
except FileNotFoundError:
    print(f"No se encontro {RUTA_TOP10_RESULTADOS} para comparar (correr primero entrenamiento_modelos_optimizado.py)")

print("\nC1 COMPLETADO")
print("=" * 60)
