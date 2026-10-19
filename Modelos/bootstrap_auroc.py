# ==========================================
# TESIS: PREDICCION DE PERDIDA GESTACIONAL
# T1: PRUEBA DE SIGNIFICANCIA ESTADISTICA DEL AUROC (BOOTSTRAP)
# EJECUCION LOCAL
# ==========================================
#
# QUE HACE ESTE SCRIPT, EN UNA FRASE:
# para cada uno de los 5 modelos ya optimizados, calcula un intervalo de
# confianza al 95% del AUROC mediante bootstrap no parametrico (B=2,000
# remuestreos del conjunto de prueba), y contrasta H0: AUROC=0.5 (el modelo
# no discrimina mejor que el azar) vs H1: AUROC>0.5.
#
# COMO FUNCIONA EL BOOTSTRAP AQUI:
# se remuestrea CON REEMPLAZO el conjunto de prueba (mismas predicciones ya
# calculadas, no se reentrena nada dentro del loop), se calcula el AUROC de
# cada remuestreo, y se repite 2,000 veces. El intervalo de confianza al 95%
# se obtiene con el metodo percentil: los percentiles 2.5 y 97.5 de esos
# 2,000 valores de AUROC. Si el limite inferior del intervalo supera 0.5, se
# rechaza H0 con ese nivel de confianza.
#
# Dependencias (instalar una sola vez desde la terminal, no aqui adentro):
#   py -m pip install pandas scikit-learn xgboost imbalanced-learn

from pathlib import Path

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTENC

RANDOM_STATE = 2023
B = 2000  # numero de remuestreos bootstrap

BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_ARCHIVO = BASE_DIR / "boruta" / "dataset_topN_importancia.csv"
RUTA_GUARDAR = BASE_DIR / "Modelos" / "bootstrap_auroc.csv"

COLUMNAS_CATEGORICAS = ['V190', 'V025', 'V106', 'V501', 'QS206']
COLUMNA_NOMINAL = ['V501']

print("=" * 60)
print("TESIS: PREDICCION DE PERDIDA GESTACIONAL")
print("T1: BOOTSTRAP DEL AUROC (H0: AUROC=0.5 vs H1: AUROC>0.5)")
print("=" * 60)

df = pd.read_csv(RUTA_ARCHIVO, encoding='utf-8-sig')
feature_names = [c for c in df.columns if c != 'target']
X = df[feature_names].copy()
y = df['target'].astype(int).copy()

categorical_features = [feature_names.index(c) for c in COLUMNAS_CATEGORICAS if c in feature_names]

# Mismo split 65/15/20 que Modelos/entrenamiento_modelos_optimizado.py
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.1875, random_state=RANDOM_STATE, stratify=y_train_full
)
print(f"\nTest (sobre el que se hace el bootstrap): {X_test.shape[0]:,} filas "
      f"({(y_test == 1).sum():,} con perdida / {(y_test == 0).sum():,} sin perdida)")

smotenc = SMOTENC(categorical_features=categorical_features, sampling_strategy='auto', random_state=RANDOM_STATE)
X_train_bal, y_train_bal = smotenc.fit_resample(X_train, y_train)

# One-Hot de V501 (fix C4), igual que en entrenamiento_modelos_optimizado.py
onehot_v501 = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
onehot_v501.fit(X_train_bal[COLUMNA_NOMINAL])
nombres_dummies_v501 = onehot_v501.get_feature_names_out(COLUMNA_NOMINAL).tolist()


def aplicar_onehot_v501(df_x):
    dummies = pd.DataFrame(
        onehot_v501.transform(df_x[COLUMNA_NOMINAL]),
        columns=nombres_dummies_v501,
        index=df_x.index,
    )
    return pd.concat([df_x.drop(columns=COLUMNA_NOMINAL), dummies], axis=1)


X_train_bal = aplicar_onehot_v501(X_train_bal)
X_test = aplicar_onehot_v501(X_test)
columnas_escalar = [c for c in X_train_bal.columns if c not in nombres_dummies_v501]

preprocesador = ColumnTransformer(transformers=[('escalado', StandardScaler(), columnas_escalar)], remainder='passthrough')

# Hiperparametros OPTIMOS ya encontrados en entrenamiento_modelos_optimizado.py
MODELOS = {
    'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=1, random_state=RANDOM_STATE, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=RANDOM_STATE),
    'XGBoost': XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1),
    'KNN': Pipeline(steps=[('preprocesar', preprocesador), ('modelo', KNeighborsClassifier(n_neighbors=11, weights='uniform'))]),
    'SVM': Pipeline(steps=[('preprocesar', preprocesador), ('modelo', SVC(kernel='rbf', C=1, gamma='auto', probability=True, random_state=RANDOM_STATE))]),
}

rng = np.random.default_rng(RANDOM_STATE)
y_test_arr = y_test.to_numpy()
n_test = len(y_test_arr)

resultados = []
print(f"\nEjecutando bootstrap (B={B}) para cada modelo...")

for nombre, modelo in MODELOS.items():
    modelo.fit(X_train_bal, y_train_bal)
    proba_test = modelo.predict_proba(X_test)[:, 1]
    auc_puntual = roc_auc_score(y_test_arr, proba_test)

    aucs_bootstrap = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n_test, n_test)
        y_b, p_b = y_test_arr[idx], proba_test[idx]
        if len(np.unique(y_b)) < 2:  # remuestreo degenerado (solo 1 clase), se recalcula
            while len(np.unique(y_b)) < 2:
                idx = rng.integers(0, n_test, n_test)
                y_b, p_b = y_test_arr[idx], proba_test[idx]
        aucs_bootstrap[b] = roc_auc_score(y_b, p_b)

    ic_inf, ic_sup = np.percentile(aucs_bootstrap, [2.5, 97.5])
    rechaza_h0 = ic_inf > 0.5

    resultados.append({
        'Modelo': nombre,
        'AUC_ROC': auc_puntual,
        'IC95_inferior': ic_inf,
        'IC95_superior': ic_sup,
        'Rechaza_H0_AUC=0.5': rechaza_h0,
    })
    print(f"\n{nombre}")
    print(f"   AUC-ROC puntual: {auc_puntual:.4f}")
    print(f"   IC 95% (bootstrap, B={B}): [{ic_inf:.4f}, {ic_sup:.4f}]")
    print(f"   H0 (AUROC=0.5) {'RECHAZADA' if rechaza_h0 else 'NO rechazada'} al 95% de confianza "
          f"({'el limite inferior supera 0.5' if rechaza_h0 else 'el intervalo incluye 0.5'})")

df_resultados = pd.DataFrame(resultados).sort_values('AUC_ROC', ascending=False).reset_index(drop=True)
df_resultados.to_csv(RUTA_GUARDAR, index=False, encoding='utf-8-sig')

print("\n" + "=" * 60)
print("TABLA FINAL (AUROC + IC 95%, ordenada de mayor a menor)")
print("=" * 60)
print(df_resultados.to_string(index=False))
print(f"\nTabla guardada en: {RUTA_GUARDAR}")

n_confirmados = df_resultados['Rechaza_H0_AUC=0.5'].sum()
print(f"\nHipotesis general: {n_confirmados} de {len(df_resultados)} modelos muestran una capacidad de "
      f"discriminacion estadisticamente significativa (IC 95% del AUROC excluye 0.5).")

print("\nT1 (BOOTSTRAP AUROC) COMPLETADO")
print("=" * 60)
