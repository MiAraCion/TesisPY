# ==========================================
# TESIS: PREDICCION DE PERDIDA GESTACIONAL
# FASE 2 (C2): VALIDACION DEL EFECTO DE SMOTE-NC EN LOS 5 MODELOS
# EJECUCION LOCAL
# ==========================================
#
# QUE HACE ESTE SCRIPT, EN UNA FRASE:
# repite comparacion_con_sin_smote.py (que solo probaba Random Forest) para
# los otros 4 modelos (XGBoost, Gradient Boosting, SVM, KNN), usando los
# hiperparametros YA OPTIMIZADOS de Modelos/entrenamiento_modelos_optimizado.py,
# para ver si el efecto mixto de SMOTE-NC (sube Recall/F1, baja AUROC/PR-AUC)
# se repite en los 5 modelos o es propio de Random Forest.
#
# POR QUE LOS HIPERPARAMETROS YA OPTIMIZADOS Y NO LOS POR DEFECTO:
# Se aisla UNA sola variable (con/sin SMOTE-NC) manteniendo fija la misma
# arquitectura ya afinada para cada modelo, en vez de mezclar el efecto del
# balanceo con el de una configuracion generica sin ajustar.
#
# Dependencias (instalar una sola vez desde la terminal, no aqui adentro):
#   py -m pip install pandas scikit-learn matplotlib xgboost imbalanced-learn

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
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
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
)
from imblearn.over_sampling import SMOTENC

RANDOM_STATE = 2023

BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_ARCHIVO = BASE_DIR / "boruta" / "dataset_topN_importancia.csv"
RUTA_GUARDAR = BASE_DIR / "Validacion_SMOTE" / "comparacion_5_modelos.csv"
RUTA_GRAFICO = BASE_DIR / "Validacion_SMOTE" / "grafico_5_modelos_con_sin_smote.png"

COLUMNAS_CATEGORICAS = ['V190', 'V025', 'V106', 'V501', 'QS206']
COLUMNA_NOMINAL = ['V501']

print("=" * 60)
print("TESIS: PREDICCION DE PERDIDA GESTACIONAL")
print("FASE 2 (C2): SMOTE-NC EN LOS 5 MODELOS (CON vs. SIN)")
print("=" * 60)

df = pd.read_csv(RUTA_ARCHIVO, encoding='utf-8-sig')
feature_names = [c for c in df.columns if c != 'target']
X = df[feature_names].copy()
y = df['target'].astype(int).copy()

categorical_features = [feature_names.index(c) for c in COLUMNAS_CATEGORICAS if c in feature_names]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain (sin tocar): {X_train.shape[0]:,} filas | Test: {X_test.shape[0]:,} filas")

smotenc = SMOTENC(categorical_features=categorical_features, sampling_strategy='auto', random_state=RANDOM_STATE)
X_train_bal, y_train_bal = smotenc.fit_resample(X_train, y_train)
print(f"Train balanceado: {X_train_bal.shape[0]:,} filas")

# One-Hot de V501 (fix C4) para KNN/SVM, consistente en sin/con SMOTE-NC.
onehot_v501 = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
onehot_v501.fit(X_train[COLUMNA_NOMINAL])
nombres_dummies_v501 = onehot_v501.get_feature_names_out(COLUMNA_NOMINAL).tolist()


def aplicar_onehot(df_x):
    dummies = pd.DataFrame(
        onehot_v501.transform(df_x[COLUMNA_NOMINAL]),
        columns=nombres_dummies_v501,
        index=df_x.index,
    )
    return pd.concat([df_x.drop(columns=COLUMNA_NOMINAL), dummies], axis=1)


X_train_oh = aplicar_onehot(X_train)
X_train_bal_oh = aplicar_onehot(X_train_bal)
X_test_oh = aplicar_onehot(X_test)
columnas_escalar = [c for c in X_train_oh.columns if c not in nombres_dummies_v501]

preprocesador = ColumnTransformer(transformers=[
    ('escalado', StandardScaler(), columnas_escalar),
], remainder='passthrough')

# Hiperparametros OPTIMOS ya encontrados en Modelos/entrenamiento_modelos_optimizado.py
CONFIGS = {
    'Random Forest': lambda: RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=1, random_state=RANDOM_STATE, n_jobs=-1),
    'Gradient Boosting': lambda: GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=RANDOM_STATE),
    'XGBoost': lambda: XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1),
    'KNN': lambda: Pipeline(steps=[('preprocesar', preprocesador), ('modelo', KNeighborsClassifier(n_neighbors=11, weights='uniform'))]),
    'SVM': lambda: Pipeline(steps=[('preprocesar', preprocesador), ('modelo', SVC(kernel='rbf', C=1, gamma='auto', probability=True, random_state=RANDOM_STATE))]),
}
USA_ONEHOT = {'Random Forest': False, 'Gradient Boosting': False, 'XGBoost': False, 'KNN': True, 'SVM': True}


def calcular_metricas(nombre, y_true, y_pred, y_proba):
    return {
        'Modelo': nombre.split(' - ')[0],
        'Condicion': nombre.split(' - ')[1],
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced_Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Precision_clase1': precision_score(y_true, y_pred, zero_division=0),
        'Recall_clase1': recall_score(y_true, y_pred, zero_division=0),
        'F1_clase1': f1_score(y_true, y_pred, zero_division=0),
        'AUC_ROC': roc_auc_score(y_true, y_proba),
        'PR_AUC': average_precision_score(y_true, y_proba),
    }


resultados = []
for nombre, constructor in CONFIGS.items():
    print(f"\n--- {nombre} ---")
    usa_oh = USA_ONEHOT[nombre]
    Xtr_sin = X_train_oh if usa_oh else X_train
    Xte = X_test_oh if usa_oh else X_test
    Xtr_con = X_train_bal_oh if usa_oh else X_train_bal

    modelo_sin = constructor()
    modelo_sin.fit(Xtr_sin, y_train)
    pred_sin = modelo_sin.predict(Xte)
    proba_sin = modelo_sin.predict_proba(Xte)[:, 1]
    fila_sin = calcular_metricas(f"{nombre} - SIN SMOTE-NC", y_test, pred_sin, proba_sin)
    resultados.append(fila_sin)
    print(f"   SIN SMOTE-NC -> Recall={fila_sin['Recall_clase1']:.4f} F1={fila_sin['F1_clase1']:.4f} AUC={fila_sin['AUC_ROC']:.4f} PR-AUC={fila_sin['PR_AUC']:.4f}")

    modelo_con = constructor()
    modelo_con.fit(Xtr_con, y_train_bal)
    pred_con = modelo_con.predict(Xte)
    proba_con = modelo_con.predict_proba(Xte)[:, 1]
    fila_con = calcular_metricas(f"{nombre} - CON SMOTE-NC", y_test, pred_con, proba_con)
    resultados.append(fila_con)
    print(f"   CON SMOTE-NC -> Recall={fila_con['Recall_clase1']:.4f} F1={fila_con['F1_clase1']:.4f} AUC={fila_con['AUC_ROC']:.4f} PR-AUC={fila_con['PR_AUC']:.4f}")

    print(f"   Delta: Recall {fila_con['Recall_clase1']-fila_sin['Recall_clase1']:+.4f} | F1 {fila_con['F1_clase1']-fila_sin['F1_clase1']:+.4f} "
          f"| AUC {fila_con['AUC_ROC']-fila_sin['AUC_ROC']:+.4f} | PR-AUC {fila_con['PR_AUC']-fila_sin['PR_AUC']:+.4f}")

df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv(RUTA_GUARDAR, index=False, encoding='utf-8-sig')
print(f"\nTabla guardada en: {RUTA_GUARDAR}")

print("\n" + "=" * 60)
print("RESUMEN: ¿EL EFECTO MIXTO SE REPITE EN LOS 5 MODELOS?")
print("=" * 60)
for nombre in CONFIGS:
    sin = df_resultados[(df_resultados['Modelo'] == nombre) & (df_resultados['Condicion'] == 'SIN SMOTE-NC')].iloc[0]
    con = df_resultados[(df_resultados['Modelo'] == nombre) & (df_resultados['Condicion'] == 'CON SMOTE-NC')].iloc[0]
    recall_sube = con['Recall_clase1'] > sin['Recall_clase1']
    f1_sube = con['F1_clase1'] > sin['F1_clase1']
    auc_baja = con['AUC_ROC'] < sin['AUC_ROC']
    prauc_baja = con['PR_AUC'] < sin['PR_AUC']
    patron = "SI (mismo patron que RF)" if (recall_sube and f1_sube and auc_baja and prauc_baja) else "NO exactamente igual"
    print(f"   {nombre}: {patron}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
modelos_lista = list(CONFIGS.keys())
x = range(len(modelos_lista))
recall_sin = [df_resultados[(df_resultados['Modelo'] == m) & (df_resultados['Condicion'] == 'SIN SMOTE-NC')]['Recall_clase1'].iloc[0] for m in modelos_lista]
recall_con = [df_resultados[(df_resultados['Modelo'] == m) & (df_resultados['Condicion'] == 'CON SMOTE-NC')]['Recall_clase1'].iloc[0] for m in modelos_lista]
auc_sin = [df_resultados[(df_resultados['Modelo'] == m) & (df_resultados['Condicion'] == 'SIN SMOTE-NC')]['AUC_ROC'].iloc[0] for m in modelos_lista]
auc_con = [df_resultados[(df_resultados['Modelo'] == m) & (df_resultados['Condicion'] == 'CON SMOTE-NC')]['AUC_ROC'].iloc[0] for m in modelos_lista]

axes[0].bar([i - 0.2 for i in x], recall_sin, width=0.4, label='Sin SMOTE-NC', color='lightcoral')
axes[0].bar([i + 0.2 for i in x], recall_con, width=0.4, label='Con SMOTE-NC', color='seagreen')
axes[0].set_xticks(list(x)); axes[0].set_xticklabels(modelos_lista, rotation=20)
axes[0].set_ylabel('Recall (clase 1)'); axes[0].set_title('Recall: efecto de SMOTE-NC'); axes[0].legend()

axes[1].bar([i - 0.2 for i in x], auc_sin, width=0.4, label='Sin SMOTE-NC', color='lightcoral')
axes[1].bar([i + 0.2 for i in x], auc_con, width=0.4, label='Con SMOTE-NC', color='seagreen')
axes[1].set_xticks(list(x)); axes[1].set_xticklabels(modelos_lista, rotation=20)
axes[1].set_ylabel('AUC-ROC'); axes[1].set_title('AUC-ROC: efecto de SMOTE-NC'); axes[1].legend()

plt.tight_layout()
plt.savefig(RUTA_GRAFICO, dpi=150)
print(f"\nGrafico guardado en: {RUTA_GRAFICO}")

print("\nC2 COMPLETADO")
print("=" * 60)
