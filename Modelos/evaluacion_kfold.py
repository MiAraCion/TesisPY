# ==========================================
# TESIS: PREDICCION DE PERDIDA GESTACIONAL
# FASE 3 (COMPLEMENTO): VALIDACION CRUZADA K-FOLD + MATRIZ DE CONFUSION
# EJECUCION LOCAL
# ==========================================
#
# QUE HACE ESTE SCRIPT, EN UNA FRASE:
# repite el benchmark de los 5 modelos (Modelos/entrenamiento_modelos.py),
# pero con validacion cruzada estratificada de 5 folds (en vez de un solo
# split train/test) y agrega F1 Macro/Ponderado y la matriz de confusion,
# que son los 3 indicadores de la Fase 3 de la matriz de consistencia que
# faltaban.
#
# POR QUE SMOTENC DENTRO DE CADA FOLD Y NO UNA VEZ AL INICIO:
# Si se balancearan las clases ANTES de partir en folds, algunas filas
# sinteticas de un fold quedarian generadas a partir de vecinos que caen en
# el fold de prueba de otra iteracion (fuga de datos entre folds). El orden
# correcto es: por cada fold, balancear SOLO la porcion de train de ESE
# fold, y evaluar contra la porcion de test de ESE fold (que nunca se
# balancea y mantiene la proporcion real ~91/9).
#
# POR QUE F1 MACRO Y PONDERADO ADEMAS DEL F1 DE LA CLASE 1:
# Modelos/entrenamiento_modelos.py solo reporta el F1 de la clase 1 (la
# que realmente interesa detectar). La matriz de consistencia pide ademas
# la version Macro (promedio simple entre clase 0 y clase 1, sin pesar por
# frecuencia) y Ponderada/Weighted (promedio pesado por cuantos casos reales
# tiene cada clase) - estas dan una vision mas completa del balance general
# del modelo, no solo de la clase minoritaria.
#
# Dependencias (instalar una sola vez desde la terminal, no aqui adentro):
#   py -m pip install pandas scikit-learn matplotlib xgboost imbalanced-learn

# ==========================================
# PASO 1: IMPORTAR LIBRERIAS
# ==========================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix,
    average_precision_score,
)
from imblearn.over_sampling import SMOTENC

RANDOM_STATE = 2023
N_FOLDS = 5

# ==========================================
# PASO 2: CONFIGURACION DE RUTAS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_ARCHIVO = BASE_DIR / "boruta" / "dataset_topN_importancia.csv"
RUTA_GUARDAR_KFOLD = BASE_DIR / "Modelos" / "resultados_kfold.csv"
RUTA_GUARDAR_MATRICES = BASE_DIR / "Modelos" / "matrices_confusion.csv"
RUTA_GRAFICO_KFOLD = BASE_DIR / "Modelos" / "grafico_kfold.png"
RUTA_GRAFICO_MATRIZ = BASE_DIR / "Modelos" / "matriz_confusion_mejor_modelo.png"

COLUMNAS_CATEGORICAS = ['V190', 'V025', 'V106', 'V501', 'QS206']
COLUMNA_NOMINAL = ['V501']  # one-hot aplicado a los 5 modelos (fix C4), dentro de cada fold

print(f"Buscando archivo en: {RUTA_ARCHIVO}")

print("\n" + "=" * 60)
print("TESIS: PREDICCION DE PERDIDA GESTACIONAL")
print(f"VALIDACION CRUZADA K-FOLD (k={N_FOLDS}) + MATRIZ DE CONFUSION")
print("=" * 60)

# ==========================================
# PASO 3: CARGAR DATOS
# ==========================================

df = pd.read_csv(RUTA_ARCHIVO, encoding='utf-8-sig')
feature_names = [c for c in df.columns if c != 'target']
X = df[feature_names].copy()
y = df['target'].astype(int).copy()

categorical_features = [feature_names.index(c) for c in COLUMNAS_CATEGORICAS if c in feature_names]

print(f"Dataset: {df.shape[0]:,} filas | Features: {feature_names}")


def aplicar_onehot_v501(encoder, nombres_dummies, df_x):
    dummies = pd.DataFrame(
        encoder.transform(df_x[COLUMNA_NOMINAL]),
        columns=nombres_dummies,
        index=df_x.index,
    )
    return pd.concat([df_x.drop(columns=COLUMNA_NOMINAL), dummies], axis=1)


# Hiperparametros: los OPTIMOS encontrados en
# Modelos/entrenamiento_modelos_optimizado.py (busqueda sobre el set de
# validacion, tras el fix C3: sin M14, con V233=0 incluido, join left en
# vez de inner), no los valores por defecto. Este k-fold valida que tan
# estable es el desempeno del modelo YA AFINADO, no de una configuracion
# generica. One-Hot de V501 (fix C4) aplicado a los 5 modelos por igual,
# no solo a KNN/SVM (ver PASO 4).
def construir_modelos(columnas_escalar):
    preprocesador = ColumnTransformer(transformers=[
        ('escalado', StandardScaler(), columnas_escalar),
    ], remainder='passthrough')
    return {
        'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=1, random_state=RANDOM_STATE, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=RANDOM_STATE),
        'XGBoost': XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1),
        'KNN': Pipeline(steps=[('preprocesar', preprocesador), ('modelo', KNeighborsClassifier(n_neighbors=11, weights='uniform'))]),
        'SVM': Pipeline(steps=[('preprocesar', preprocesador), ('modelo', SVC(kernel='rbf', C=1, gamma='auto', probability=True, random_state=RANDOM_STATE))]),
    }


# ==========================================
# PASO 4: VALIDACION CRUZADA ESTRATIFICADA (SMOTENC dentro de cada fold)
# ==========================================

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

resultados_por_fold = []
matrices_acumuladas = {}  # suma de las matrices de confusion de los 5 folds, por modelo

print(f"\nEjecutando validacion cruzada de {N_FOLDS} folds para cada modelo...")

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
    y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

    smotenc = SMOTENC(categorical_features=categorical_features, sampling_strategy='auto', random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smotenc.fit_resample(X_train_fold, y_train_fold)

    # One-Hot de V501 (fix C4), ajustado sobre el train balanceado de ESTE
    # fold y aplicado tanto al train como al test del mismo fold.
    onehot_v501 = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    onehot_v501.fit(X_train_bal[COLUMNA_NOMINAL])
    nombres_dummies_v501 = onehot_v501.get_feature_names_out(COLUMNA_NOMINAL).tolist()

    X_train_bal = aplicar_onehot_v501(onehot_v501, nombres_dummies_v501, X_train_bal)
    X_test_fold = aplicar_onehot_v501(onehot_v501, nombres_dummies_v501, X_test_fold)
    columnas_escalar = [c for c in X_train_bal.columns if c not in nombres_dummies_v501]

    modelos = construir_modelos(columnas_escalar)
    for nombre, modelo in modelos.items():
        modelo.fit(X_train_bal, y_train_bal)
        pred = modelo.predict(X_test_fold)
        proba = modelo.predict_proba(X_test_fold)[:, 1]

        resultados_por_fold.append({
            'Fold': fold_idx,
            'Modelo': nombre,
            'Accuracy': accuracy_score(y_test_fold, pred),
            'Balanced_Accuracy': balanced_accuracy_score(y_test_fold, pred),
            'Precision_clase1': precision_score(y_test_fold, pred, zero_division=0),
            'Recall_clase1': recall_score(y_test_fold, pred, zero_division=0),
            'F1_clase1': f1_score(y_test_fold, pred, zero_division=0),
            'F1_macro': f1_score(y_test_fold, pred, average='macro', zero_division=0),
            'F1_ponderado': f1_score(y_test_fold, pred, average='weighted', zero_division=0),
            'AUC_ROC': roc_auc_score(y_test_fold, proba),
            'PR_AUC': average_precision_score(y_test_fold, proba),
        })

        cm = confusion_matrix(y_test_fold, pred, labels=[0, 1])
        if nombre not in matrices_acumuladas:
            matrices_acumuladas[nombre] = cm
        else:
            matrices_acumuladas[nombre] = matrices_acumuladas[nombre] + cm

    print(f"   Fold {fold_idx}/{N_FOLDS} completado "
          f"(train: {len(train_idx):,} filas antes de balancear -> {X_train_bal.shape[0]:,} balanceadas | test: {len(test_idx):,} filas)")

df_folds = pd.DataFrame(resultados_por_fold)

# ==========================================
# PASO 5: RESUMEN (media +/- desviacion estandar entre los 5 folds)
# ==========================================

print("\n" + "=" * 60)
print(f"RESUMEN K-FOLD (media +/- desv. estandar de los {N_FOLDS} folds)")
print("=" * 60)

metricas_cols = ['Accuracy', 'Balanced_Accuracy', 'Precision_clase1', 'Recall_clase1',
                  'F1_clase1', 'F1_macro', 'F1_ponderado', 'AUC_ROC', 'PR_AUC']

resumen = df_folds.groupby('Modelo')[metricas_cols].agg(['mean', 'std'])
resumen.columns = [f"{col}_{stat}" for col, stat in resumen.columns]
resumen = resumen.reset_index().sort_values('F1_clase1_mean', ascending=False).reset_index(drop=True)
resumen.insert(0, 'Posicion', range(1, len(resumen) + 1))

for _, fila in resumen.iterrows():
    print(f"\n{fila['Modelo']}")
    print(f"   AUC-ROC:      {fila['AUC_ROC_mean']:.4f} +/- {fila['AUC_ROC_std']:.4f}")
    print(f"   F1 (clase 1): {fila['F1_clase1_mean']:.4f} +/- {fila['F1_clase1_std']:.4f}")
    print(f"   F1 Macro:     {fila['F1_macro_mean']:.4f} +/- {fila['F1_macro_std']:.4f}")
    print(f"   F1 Ponderado: {fila['F1_ponderado_mean']:.4f} +/- {fila['F1_ponderado_std']:.4f}")
    print(f"   Recall:       {fila['Recall_clase1_mean']:.4f} +/- {fila['Recall_clase1_std']:.4f}")

df_folds.to_csv(RUTA_GUARDAR_KFOLD.with_name('resultados_kfold_por_fold.csv'), index=False, encoding='utf-8-sig')
resumen.to_csv(RUTA_GUARDAR_KFOLD, index=False, encoding='utf-8-sig')
print(f"\nResultados por fold guardados en: {RUTA_GUARDAR_KFOLD.with_name('resultados_kfold_por_fold.csv')}")
print(f"Resumen (media +/- std) guardado en: {RUTA_GUARDAR_KFOLD}")

# ==========================================
# PASO 6: MATRICES DE CONFUSION ACUMULADAS (suma de los 5 folds)
# ==========================================
# Sumar las matrices de los 5 folds da una matriz de confusion sobre el
# dataset COMPLETO (cada fila paso por el test set exactamente una vez a lo
# largo de las 5 iteraciones), mas robusta que una matriz de un solo split.

print("\n" + "-" * 60)
print("MATRICES DE CONFUSION ACUMULADAS (suma de los 5 folds)")
print("-" * 60)

filas_matrices = []
for nombre, cm in matrices_acumuladas.items():
    tn, fp, fn, tp = cm.ravel()
    print(f"\n{nombre}")
    print(f"   [[TN={tn:>5}  FP={fp:>5}]")
    print(f"    [FN={fn:>5}  TP={tp:>5}]]")
    filas_matrices.append({'Modelo': nombre, 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp})

df_matrices = pd.DataFrame(filas_matrices)
df_matrices.to_csv(RUTA_GUARDAR_MATRICES, index=False, encoding='utf-8-sig')
print(f"\nMatrices de confusion guardadas en: {RUTA_GUARDAR_MATRICES}")

# ==========================================
# PASO 7: GRAFICOS
# ==========================================

plt.figure(figsize=(9, 5))
orden = resumen.iloc[::-1]
plt.barh(orden['Modelo'], orden['F1_clase1_mean'], xerr=orden['F1_clase1_std'], color='steelblue', capsize=4)
plt.xlabel('F1-score (clase 1), media +/- desv. estandar (5-fold CV)')
plt.title(f'Validacion cruzada estratificada ({N_FOLDS} folds)\nSMOTENC aplicado dentro de cada fold')
plt.tight_layout()
plt.savefig(RUTA_GRAFICO_KFOLD, dpi=150)
print(f"Grafico guardado en: {RUTA_GRAFICO_KFOLD}")

mejor_modelo = resumen.iloc[0]['Modelo']
cm_mejor = matrices_acumuladas[mejor_modelo]
plt.figure(figsize=(5, 4.5))
plt.imshow(cm_mejor, cmap='Blues')
for i in range(2):
    for j in range(2):
        plt.text(j, i, f"{cm_mejor[i, j]:,}", ha='center', va='center',
                  color='white' if cm_mejor[i, j] > cm_mejor.max() / 2 else 'black', fontsize=14)
plt.xticks([0, 1], ['Pred. 0 (sin perdida)', 'Pred. 1 (perdida)'])
plt.yticks([0, 1], ['Real 0', 'Real 1'])
plt.title(f'Matriz de confusion acumulada (5-fold)\nMejor modelo: {mejor_modelo}')
plt.tight_layout()
plt.savefig(RUTA_GRAFICO_MATRIZ, dpi=150)
print(f"Matriz de confusion del mejor modelo guardada en: {RUTA_GRAFICO_MATRIZ}")

print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
print(f"\nMejor modelo (5-fold CV, por F1 clase 1): {mejor_modelo}")
print(f"   F1: {resumen.iloc[0]['F1_clase1_mean']:.4f} +/- {resumen.iloc[0]['F1_clase1_std']:.4f}")
print(f"   AUC-ROC: {resumen.iloc[0]['AUC_ROC_mean']:.4f} +/- {resumen.iloc[0]['AUC_ROC_std']:.4f}")

print("\nEVALUACION K-FOLD COMPLETADA")
print("=" * 60)
