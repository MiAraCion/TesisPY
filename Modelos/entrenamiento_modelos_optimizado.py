# ==========================================
# TESIS: PREDICCION DE PERDIDA GESTACIONAL
# ENTRENAMIENTO OPTIMIZADO: HIPERPARAMETROS + UMBRAL DE DECISION
# EJECUCION LOCAL
# ==========================================
#
# QUE HACE ESTE SCRIPT, EN UNA FRASE:
# repite el entrenamiento de los 5 modelos, pero esta vez (1) prueba varias
# combinaciones de hiperparametros por modelo y se queda con la mejor, y
# (2) en vez de usar el umbral por defecto (0.5) para decidir "clase 1",
# busca el umbral que maximiza F1 en un conjunto de validacion separado.
#
# POR QUE HACE FALTA UN TERCER CONJUNTO (VALIDACION), ADEMAS DE TRAIN/TEST:
# Si se eligieran los hiperparametros O el umbral mirando el mismo test que
# despues se usa para reportar el resultado final, el numero final quedaria
# artificialmente inflado (el modelo/umbral se habria "ajustado" al test).
# Por eso este script parte el dataset en TRES conjuntos:
#   - train (55% aprox.): se balancea con SMOTENC, se usa para entrenar.
#   - validacion (15% aprox., proporcion real 91/9, SIN balancear): se usa
#     SOLO para elegir hiperparametros y el umbral de decision.
#   - test (30%, proporcion real 91/9, SIN balancear): se usa SOLO al
#     final, una vez, para reportar el resultado. Nunca participa en
#     ninguna decision de ajuste.
# El split train/test (70/30) es identico al de SMOTENC/smotenc_balanceo.py
# (mismo random_state, mismo test_size, mismos datos de origen), asi que el
# conjunto de test de este script coincide exactamente con
# SMOTENC/test_original.csv. Lo que cambia es que el 70% de train se
# subdivide ademas en train+validacion.
#
# POR QUE BUSQUEDA "MANUAL" SOBRE VALIDACION EN VEZ DE GridSearchCV:
# GridSearchCV/RandomizedSearchCV hacen validacion cruzada (K-fold) DENTRO
# del train, lo cual multiplica el tiempo de computo por K en cada
# candidato. Como ya se separo un conjunto de validacion real, alcanza con
# entrenar cada candidato una vez sobre train y medir su F1 en validacion:
# mismo objetivo, una fraccion del computo. Para SVM (el mas lento de
# entrenar) se usa ademas una submuestra del train balanceado solo durante
# la busqueda de hiperparametros, por costo computacional; el modelo final
# elegido se reentrena sobre el train balanceado COMPLETO.
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

from sklearn.model_selection import train_test_split, ParameterSampler
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
    precision_recall_curve,
)
from imblearn.over_sampling import SMOTENC

# ==========================================
# PASO 2: CONFIGURACION DE RUTAS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_ARCHIVO = BASE_DIR / "boruta" / "dataset_topN_importancia.csv"
RUTA_GUARDAR = BASE_DIR / "Modelos" / "resultados_modelos_optimizado.csv"
RUTA_GRAFICO = BASE_DIR / "Modelos" / "grafico_comparacion_optimizado.png"

COLUMNAS_CATEGORICAS = ['V190', 'V025', 'V106', 'V501']  # para SMOTENC
COLUMNA_NOMINAL = ['V501']  # para one-hot en KNN/SVM
RANDOM_STATE = 2023
LIMITE_SVM_BUSQUEDA = 8000  # submuestra solo para la busqueda de hiperparametros de SVM

print(f"Buscando archivo en: {RUTA_ARCHIVO}")

print("\n" + "=" * 60)
print("TESIS: PREDICCION DE PERDIDA GESTACIONAL")
print("ENTRENAMIENTO OPTIMIZADO (HIPERPARAMETROS + UMBRAL)")
print("=" * 60)

df = pd.read_csv(RUTA_ARCHIVO, encoding='utf-8-sig')
feature_names = [c for c in df.columns if c != 'target']
X = df[feature_names].copy()
y = df['target'].astype(int).copy()

categorical_features = [feature_names.index(c) for c in COLUMNAS_CATEGORICAS if c in feature_names]
columnas_escalar = [c for c in feature_names if c not in COLUMNA_NOMINAL]

print(f"Dataset: {df.shape[0]:,} filas | Features: {feature_names}")

# ==========================================
# PASO 4: SPLIT EN TRES PARTES (train / validacion / test)
# ==========================================
# El split externo train_full/test es 80/20 (test_size=0.20), siguiendo el
# protocolo del marco teorico. La validacion se sigue reservando aparte,
# como el 15% del TOTAL: como ahora train_full es el 80% del total, el
# test_size interno para lograr ese 15% es 0.15/0.80 = 0.1875 (en vez del
# 0.2143 que correspondia cuando train_full era 70%). Proporciones
# finales: train=65%, validacion=15%, test=20%.

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.1875, random_state=RANDOM_STATE, stratify=y_train_full
)

print(f"\nTrain (para entrenar, se balanceara con SMOTENC): {X_train.shape[0]:,} filas")
print(f"Validacion (para elegir hiperparametros y umbral, proporcion real): {X_val.shape[0]:,} filas "
      f"({(y_val == 1).mean() * 100:.1f}% clase 1)")
print(f"Test (SOLO para el resultado final, proporcion real): {X_test.shape[0]:,} filas "
      f"({(y_test == 1).mean() * 100:.1f}% clase 1)")

# ==========================================
# PASO 5: SMOTENC - BALANCEO SOLO DEL TRAIN
# ==========================================

smotenc = SMOTENC(categorical_features=categorical_features, sampling_strategy='auto', random_state=RANDOM_STATE)
X_train_bal, y_train_bal = smotenc.fit_resample(X_train, y_train)

print(f"\nTrain balanceado por SMOTENC: {X_train_bal.shape[0]:,} filas "
      f"({(y_train_bal == 0).sum():,} clase 0 / {(y_train_bal == 1).sum():,} clase 1)")

preprocesador = ColumnTransformer(transformers=[
    ('escalado', StandardScaler(), columnas_escalar),
    ('onehot', OneHotEncoder(handle_unknown='ignore'), COLUMNA_NOMINAL),
])

# ==========================================
# PASO 7: ESPACIOS DE BUSQUEDA DE HIPERPARAMETROS
# ==========================================

espacios = {
    'Random Forest': {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [5, 8, 12, None],
        'min_samples_leaf': [1, 3, 5],
    },
    'Gradient Boosting': {
        'n_estimators': [100, 150, 200],
        'max_depth': [2, 3, 4],
        'learning_rate': [0.03, 0.05, 0.1],
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.03, 0.05, 0.1, 0.2],
    },
    'KNN': {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
    },
    'SVM': {
        'C': [0.5, 1, 3, 5],
        'gamma': ['scale', 'auto'],
    },
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
        return Pipeline(steps=[('preprocesar', preprocesador),
                                ('modelo', SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE, **params))])
    raise ValueError(nombre)


def mejor_umbral(y_true, proba):
    precisiones, recalls, umbrales = precision_recall_curve(y_true, proba)
    f1s = np.where((precisiones + recalls) > 0, 2 * precisiones * recalls / (precisiones + recalls + 1e-12), 0)
    mejor_idx = np.argmax(f1s[:-1]) if len(umbrales) > 0 else 0
    return umbrales[mejor_idx] if len(umbrales) > 0 else 0.5


# ==========================================
# PASO 8: BUSQUEDA DE HIPERPARAMETROS + UMBRAL, POR MODELO
# ==========================================

print("\n" + "=" * 60)
print("BUSQUEDA DE HIPERPARAMETROS (evaluada en el set de VALIDACION)")
print("=" * 60)

resultados = []
modelos_finales = {}

for nombre, espacio in espacios.items():
    print(f"\n--- {nombre} ---")
    candidatos = list(ParameterSampler(espacio, n_iter=n_iter_por_modelo[nombre], random_state=RANDOM_STATE))

    if nombre == 'SVM' and X_train_bal.shape[0] > LIMITE_SVM_BUSQUEDA:
        X_busqueda, _, y_busqueda, _ = train_test_split(
            X_train_bal, y_train_bal, train_size=LIMITE_SVM_BUSQUEDA,
            random_state=RANDOM_STATE, stratify=y_train_bal,
        )
    else:
        X_busqueda, y_busqueda = X_train_bal, y_train_bal

    mejor_f1_val = -1
    mejor_params = None
    for params in candidatos:
        modelo = construir_modelo(nombre, params)
        modelo.fit(X_busqueda, y_busqueda)
        pred_val = modelo.predict(X_val)
        f1_val = f1_score(y_val, pred_val, zero_division=0)
        print(f"   {params} -> F1 validacion: {f1_val:.4f}")
        if f1_val > mejor_f1_val:
            mejor_f1_val = f1_val
            mejor_params = params

    print(f"   Mejores hiperparametros: {mejor_params} (F1 validacion: {mejor_f1_val:.4f})")

    modelo_final = construir_modelo(nombre, mejor_params)
    modelo_final.fit(X_train_bal, y_train_bal)

    proba_val = modelo_final.predict_proba(X_val)[:, 1]
    umbral = mejor_umbral(y_val, proba_val)
    print(f"   Umbral optimo (elegido en validacion): {umbral:.4f}")

    modelos_finales[nombre] = (modelo_final, mejor_params, umbral)

# ==========================================
# PASO 9: EVALUACION FINAL SOBRE TEST (una sola vez, sin mas ajustes)
# ==========================================

print("\n" + "=" * 60)
print("EVALUACION FINAL SOBRE TEST (umbral 0.5 vs. umbral optimizado)")
print("=" * 60)

for nombre, (modelo_final, mejor_params, umbral) in modelos_finales.items():
    proba_test = modelo_final.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba_test)

    pred_05 = (proba_test >= 0.5).astype(int)
    pred_umbral = (proba_test >= umbral).astype(int)

    fila = {
        'Modelo': nombre,
        'Mejores_hiperparametros': str(mejor_params),
        'Umbral_optimo': round(umbral, 4),
        'AUC_ROC': auc,
        'F1_umbral_0.5': f1_score(y_test, pred_05, zero_division=0),
        'Recall_umbral_0.5': recall_score(y_test, pred_05, zero_division=0),
        'Precision_umbral_0.5': precision_score(y_test, pred_05, zero_division=0),
        'F1_umbral_optimo': f1_score(y_test, pred_umbral, zero_division=0),
        'Recall_umbral_optimo': recall_score(y_test, pred_umbral, zero_division=0),
        'Precision_umbral_optimo': precision_score(y_test, pred_umbral, zero_division=0),
        'Accuracy_umbral_optimo': accuracy_score(y_test, pred_umbral),
        'Balanced_Accuracy_umbral_optimo': balanced_accuracy_score(y_test, pred_umbral),
    }
    resultados.append(fila)

    print(f"\n{nombre}")
    print(f"   AUC-ROC:                    {auc:.4f}")
    print(f"   F1 con umbral 0.5:          {fila['F1_umbral_0.5']:.4f}")
    print(f"   F1 con umbral optimizado:   {fila['F1_umbral_optimo']:.4f}  (umbral={umbral:.4f})")
    print(f"   Precision/Recall optimizado: {fila['Precision_umbral_optimo']:.4f} / {fila['Recall_umbral_optimo']:.4f}")

# ==========================================
# PASO 10: TABLA COMPARATIVA, GRAFICO Y GUARDADO
# ==========================================

df_resultados = pd.DataFrame(resultados).sort_values('F1_umbral_optimo', ascending=False).reset_index(drop=True)
df_resultados.insert(0, 'Posicion', range(1, len(df_resultados) + 1))

print("\n" + "-" * 60)
print("TABLA COMPARATIVA FINAL (ordenada por F1 con umbral optimizado)")
print("-" * 60)
print(df_resultados[['Posicion', 'Modelo', 'AUC_ROC', 'F1_umbral_0.5', 'F1_umbral_optimo',
                      'Precision_umbral_optimo', 'Recall_umbral_optimo']].to_string(index=False))

plt.figure(figsize=(9, 5))
orden_grafico = df_resultados.iloc[::-1]
x = np.arange(len(orden_grafico))
plt.barh(x - 0.2, orden_grafico['F1_umbral_0.5'], height=0.4, label='Umbral 0.5', color='lightgray')
plt.barh(x + 0.2, orden_grafico['F1_umbral_optimo'], height=0.4, label='Umbral optimizado', color='steelblue')
plt.yticks(x, orden_grafico['Modelo'])
plt.xlabel('F1-score (clase 1: perdida gestacional)')
plt.title('Efecto del ajuste de hiperparametros + umbral\n(evaluado sobre test, sin tocar durante el ajuste)')
plt.legend()
plt.tight_layout()
plt.savefig(RUTA_GRAFICO, dpi=150)
print(f"\nGrafico guardado en: {RUTA_GRAFICO}")

df_resultados.to_csv(RUTA_GUARDAR, index=False, encoding='utf-8-sig')
print(f"Tabla de resultados guardada en: {RUTA_GUARDAR}")

print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
mejor = df_resultados.iloc[0]
print(f"\nMejor modelo: {mejor['Modelo']}")
print(f"   Hiperparametros: {mejor['Mejores_hiperparametros']}")
print(f"   Umbral optimo: {mejor['Umbral_optimo']}")
print(f"   F1 (umbral 0.5 -> optimizado): {mejor['F1_umbral_0.5']:.4f} -> {mejor['F1_umbral_optimo']:.4f}")
print(f"   AUC-ROC: {mejor['AUC_ROC']:.4f}")

print("\nENTRENAMIENTO OPTIMIZADO COMPLETADO")
print("=" * 60)
