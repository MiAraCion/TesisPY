# ==========================================
# TESIS: PREDICCION DE PERDIDA GESTACIONAL
# FASE 5: EXPLICABILIDAD LOCAL (LIME + SHAP Waterfall)
# EJECUCION LOCAL
# ==========================================
#
# QUE HACE ESTE SCRIPT, EN UNA FRASE:
# toma el mismo modelo ganador (Random Forest) de XAI_Global/shap_global.py
# y explica, UNA MUJER A LA VEZ, por que el modelo le asigno el nivel de
# riesgo que le asigno - a diferencia de la Fase 4 (explicabilidad GLOBAL,
# que promedia sobre todo el test set), aqui se mira caso por caso.
#
# LIME vs. SHAP LOCAL, EN UNA FRASE:
# LIME genera datos sinteticos ALREDEDOR de la mujer especifica, entrena un
# modelo simple (lineal) solo con esos datos vecinos, y usa los
# coeficientes de ese modelo simple como aproximacion de que variables
# importaron para ESA prediccion. SHAP Waterfall, en cambio, descompone la
# prediccion real del modelo complejo (no una aproximacion lineal) en la
# suma exacta de la contribucion de cada variable, partiendo del promedio
# del modelo (valor base) hasta llegar a la prediccion final de esa mujer.
#
# QUE CASOS SE ELIGEN Y POR QUE:
# se seleccionan 3 perfiles del test set (nunca visto durante el
# entrenamiento) que ilustran situaciones clinicamente distintas:
#   1) Alto riesgo bien detectado (verdadero positivo, probabilidad alta)
#   2) Bajo riesgo bien detectado (verdadero negativo, probabilidad baja)
#   3) Caso limite (probabilidad cercana a 0.5, el mas dificil de explicar)
#
# Dependencias (instalar una sola vez desde la terminal, no aqui adentro):
#   py -m pip install shap lime pandas scikit-learn matplotlib

# ==========================================
# PASO 1: IMPORTAR LIBRERIAS
# ==========================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import shap
from lime.lime_tabular import LimeTabularExplainer

RANDOM_STATE = 2023

# ==========================================
# PASO 2: CONFIGURACION DE RUTAS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_TRAIN = BASE_DIR / "SMOTENC" / "train_balanceado.csv"
RUTA_TEST = BASE_DIR / "SMOTENC" / "test_original.csv"
RUTA_SALIDA = BASE_DIR / "XAI_Local"
RUTA_RESUMEN = RUTA_SALIDA / "resumen_casos_explicados.csv"

COLUMNAS_CATEGORICAS = ['V190', 'V025', 'V106', 'QS206']  # V501 se agrega tras el One-Hot (fix C4)

print(f"Buscando archivos en: {RUTA_TRAIN.parent}")

print("\n" + "=" * 60)
print("TESIS: PREDICCION DE PERDIDA GESTACIONAL")
print("FASE 5: EXPLICABILIDAD LOCAL (LIME + SHAP Waterfall)")
print("=" * 60)

# ==========================================
# PASO 3: CARGAR DATOS Y ENTRENAR EL MISMO MODELO GANADOR
# ==========================================
# Mismos hiperparametros y semilla que en XAI_Global/shap_global.py, para
# que las explicaciones locales sean consistentes con la explicacion global
# (es el mismo modelo, solo se mira distinto nivel de detalle).

df_train = pd.read_csv(RUTA_TRAIN, encoding='utf-8-sig')
df_test = pd.read_csv(RUTA_TEST, encoding='utf-8-sig')

feature_names = [c for c in df_train.columns if c != 'target']
X_train = df_train[feature_names]
y_train = df_train['target'].astype(int)
X_test = df_test[feature_names].reset_index(drop=True)
y_test = df_test['target'].astype(int).reset_index(drop=True)

# One-Hot de V501 (fix C4), igual que en XAI_Global/shap_global.py y en
# Modelos/entrenamiento_modelos_optimizado.py: se explica el mismo modelo
# que realmente se entrena y evalua ahi.
onehot_v501 = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
onehot_v501.fit(X_train[['V501']])
nombres_dummies_v501 = onehot_v501.get_feature_names_out(['V501']).tolist()


def aplicar_onehot_v501(df_x):
    dummies = pd.DataFrame(
        onehot_v501.transform(df_x[['V501']]),
        columns=nombres_dummies_v501,
        index=df_x.index,
    )
    return pd.concat([df_x.drop(columns=['V501']), dummies], axis=1)


X_train = aplicar_onehot_v501(X_train)
X_test = aplicar_onehot_v501(X_test)
feature_names = list(X_train.columns)
COLUMNAS_CATEGORICAS = COLUMNAS_CATEGORICAS + nombres_dummies_v501
print(f"V501 codificado con One-Hot ({len(nombres_dummies_v501)} columnas): {nombres_dummies_v501}")

# Hiperparametros OPTIMOS (iguales a XAI_Global/shap_global.py, tras el fix
# C3), no los valores por defecto.
modelo = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=1, random_state=RANDOM_STATE, n_jobs=-1)
modelo.fit(X_train, y_train)
print(f"\nModelo (Random Forest, hiperparametros optimos) entrenado sobre {X_train.shape[0]:,} filas balanceadas")

proba_test = modelo.predict_proba(X_test)[:, 1]
pred_test = modelo.predict(X_test)

# ==========================================
# PASO 4: SELECCIONAR 3 CASOS REPRESENTATIVOS DEL TEST SET
# ==========================================

print("\n" + "-" * 60)
print("SELECCIONANDO CASOS REPRESENTATIVOS")
print("-" * 60)

df_test_anotado = X_test.copy()
df_test_anotado['target_real'] = y_test
df_test_anotado['prediccion'] = pred_test
df_test_anotado['probabilidad'] = proba_test

# Caso 1: verdadero positivo con la probabilidad mas alta (riesgo bien detectado)
vp = df_test_anotado[(df_test_anotado['target_real'] == 1) & (df_test_anotado['prediccion'] == 1)]
idx_alto_riesgo = vp['probabilidad'].idxmax() if len(vp) > 0 else df_test_anotado['probabilidad'].idxmax()

# Caso 2: verdadero negativo con la probabilidad mas baja (bajo riesgo bien detectado)
vn = df_test_anotado[(df_test_anotado['target_real'] == 0) & (df_test_anotado['prediccion'] == 0)]
idx_bajo_riesgo = vn['probabilidad'].idxmin() if len(vn) > 0 else df_test_anotado['probabilidad'].idxmin()

# Caso 3: caso limite, probabilidad mas cercana a 0.5 (el mas dificil de explicar)
idx_limite = (df_test_anotado['probabilidad'] - 0.5).abs().idxmin()

casos = {
    'Alto riesgo (verdadero positivo)': idx_alto_riesgo,
    'Bajo riesgo (verdadero negativo)': idx_bajo_riesgo,
    'Caso limite (probabilidad ~0.5)': idx_limite,
}

for nombre, idx in casos.items():
    fila = df_test_anotado.loc[idx]
    print(f"\n{nombre} (fila #{idx}):")
    print(f"   Target real: {fila['target_real']} | Prediccion: {fila['prediccion']} | Probabilidad: {fila['probabilidad']:.4f}")
    print(f"   Valores: {dict(fila[feature_names])}")

# ==========================================
# PASO 5: EXPLICABILIDAD LOCAL CON LIME
# ==========================================
# LimeTabularExplainer necesita el train set como referencia para saber la
# distribucion real de cada variable (para poder generar los vecinos
# sinteticos de forma realista). categorical_features indica que columnas
# NO deben tratarse como continuas al perturbar los datos.

print("\n" + "=" * 60)
print("EXPLICABILIDAD LOCAL: LIME")
print("=" * 60)

categorical_features_idx = [feature_names.index(c) for c in COLUMNAS_CATEGORICAS if c in feature_names]

explainer_lime = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    class_names=['Sin perdida', 'Perdida gestacional'],
    categorical_features=categorical_features_idx,
    mode='classification',
    random_state=RANDOM_STATE,
)

resumen_filas = []

for nombre, idx in casos.items():
    print(f"\n--- {nombre} ---")
    instancia = X_test.loc[idx].values
    explicacion_lime = explainer_lime.explain_instance(
        instancia, modelo.predict_proba, num_features=len(feature_names),
    )

    print("Contribuciones LIME (variable: peso hacia el riesgo):")
    for variable, peso in explicacion_lime.as_list():
        print(f"   {variable}: {peso:+.4f}")
        resumen_filas.append({
            'Caso': nombre, 'Metodo': 'LIME', 'Variable_condicion': variable, 'Contribucion': peso,
        })

    nombre_archivo = nombre.split(' (')[0].strip().replace(' ', '_').lower()
    fig = explicacion_lime.as_pyplot_figure()
    fig.set_size_inches(8, 5)
    plt.title(f"LIME - {nombre}")
    plt.tight_layout()
    plt.savefig(RUTA_SALIDA / f"lime_{nombre_archivo}.png", dpi=150)
    plt.close(fig)
    print(f"   Grafico guardado: lime_{nombre_archivo}.png")

# ==========================================
# PASO 6: EXPLICABILIDAD LOCAL CON SHAP WATERFALL
# ==========================================
# Waterfall muestra, para UNA fila, como se llega desde el "valor base"
# (la probabilidad promedio del modelo sobre todo el train) hasta la
# probabilidad final predicha para esa mujer, sumando el aporte (positivo o
# negativo) de cada variable.

print("\n" + "=" * 60)
print("EXPLICABILIDAD LOCAL: SHAP WATERFALL")
print("=" * 60)

explainer_shap = shap.TreeExplainer(modelo)
explicacion_shap_test = explainer_shap(X_test)

valores_shap_test = explicacion_shap_test.values
base_values = explicacion_shap_test.base_values
if valores_shap_test.ndim == 3:
    valores_shap_test = valores_shap_test[:, :, 1]
    base_values = base_values[:, 1] if np.array(base_values).ndim > 1 else base_values

for nombre, idx in casos.items():
    print(f"\n--- {nombre} ---")
    print(f"   Valor base (promedio del modelo): {base_values[idx]:.4f}")
    print(f"   Probabilidad final predicha: {proba_test[idx]:.4f}")

    explicacion_individual = shap.Explanation(
        values=valores_shap_test[idx],
        base_values=base_values[idx],
        data=X_test.loc[idx].values,
        feature_names=feature_names,
    )

    for feat, val_shap in zip(feature_names, valores_shap_test[idx]):
        print(f"   {feat}: {val_shap:+.4f}")
        resumen_filas.append({
            'Caso': nombre, 'Metodo': 'SHAP', 'Variable_condicion': feat, 'Contribucion': val_shap,
        })

    plt.figure(figsize=(8, 5))
    shap.plots.waterfall(explicacion_individual, show=False)
    nombre_archivo = nombre.split(' (')[0].strip().replace(' ', '_').lower()
    plt.title(f"SHAP Waterfall - {nombre}")
    plt.tight_layout()
    plt.savefig(RUTA_SALIDA / f"shap_waterfall_{nombre_archivo}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Grafico guardado: shap_waterfall_{nombre_archivo}.png")

# ==========================================
# PASO 7: GUARDAR RESUMEN
# ==========================================

df_resumen = pd.DataFrame(resumen_filas)
df_resumen.to_csv(RUTA_RESUMEN, index=False, encoding='utf-8-sig')
print(f"\nResumen de todas las contribuciones guardado en: {RUTA_RESUMEN}")

print("\n" + "=" * 60)
print("EXPLICABILIDAD LOCAL (LIME + SHAP WATERFALL) COMPLETADA")
print("=" * 60)
