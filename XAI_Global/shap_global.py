# ==========================================
# TESIS: PREDICCION DE PERDIDA GESTACIONAL
# FASE 4: EXPLICABILIDAD GLOBAL (TreeSHAP)
# EJECUCION LOCAL
# ==========================================
#
# QUE HACE ESTE SCRIPT, EN UNA FRASE:
# entrena el modelo ganador (Random Forest, el mas consistente en todo el
# proyecto) sobre el train balanceado, y usa TreeSHAP para explicar, a
# nivel GLOBAL (sobre todo el test set), que tan importante es cada
# variable y en que direccion empuja el riesgo de perdida gestacional.
#
# QUE ES SHAP, EN UNA FRASE (para quien lea esto sin contexto previo):
# SHAP le asigna a cada variable, para cada prediccion individual, un
# "credito" (positivo o negativo) que representa cuanto esa variable
# empujo la prediccion hacia arriba o hacia abajo respecto al promedio del
# modelo. TreeSHAP es una version exacta y rapida de SHAP disenada
# especificamente para modelos de arboles (Random Forest, XGBoost,
# Gradient Boosting), a diferencia de la version generica de SHAP que es
# mas lenta y aproximada.
#
# EXPLICABILIDAD GLOBAL vs. LOCAL (Fase 4 vs. Fase 5):
# Este script promedia/agrega los valores SHAP de TODAS las filas del test
# set para responder "en general, que variables importan mas y en que
# sentido". La Fase 5 (carpeta XAI_Local/) hace lo opuesto: mira UNA
# prediccion a la vez, para explicar un caso especifico.
#
# Dependencias (instalar una sola vez desde la terminal, no aqui adentro):
#   py -m pip install shap pandas scikit-learn matplotlib

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

RANDOM_STATE = 2023

# ==========================================
# PASO 2: CONFIGURACION DE RUTAS
# ==========================================
# Este script vive en XAI_Global/, un nivel por debajo de la raiz del
# proyecto. Reutiliza el train/test que ya genero SMOTENC/smotenc_balanceo.py
# (no se vuelve a tocar ese script).

BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_TRAIN = BASE_DIR / "SMOTENC" / "train_balanceado.csv"
RUTA_TEST = BASE_DIR / "SMOTENC" / "test_original.csv"
RUTA_GUARDAR = BASE_DIR / "XAI_Global" / "shap_importancia_global.csv"
RUTA_GUARDAR_DIMENSION = BASE_DIR / "XAI_Global" / "shap_dimension_salud_vs_sociodemografico.csv"
RUTA_BEESWARM = BASE_DIR / "XAI_Global" / "shap_beeswarm.png"
RUTA_BARRAS = BASE_DIR / "XAI_Global" / "shap_barras_importancia.png"

# Clasificacion de cada variable del top 10 en su dimension teorica, segun
# la matriz de consistencia y operacionalizacion del proyecto. Actualizado
# tras el fix C3 (M14 fuera del top 10, entran V012 y V201; ver
# boruta/resultados_boruta.csv). V501 se expande en columnas V501_x por el
# One-Hot (fix C4): se clasifican todas bajo el mismo prefijo.
DIMENSION_VARIABLE = {
    'V012': 'Sociodemografico', 'V106': 'Sociodemografico', 'V025': 'Sociodemografico',
    'V190': 'Sociodemografico', 'V040': 'Sociodemografico',
    'IMC': 'Salud', 'V201': 'Salud', 'V222': 'Salud', 'QS206': 'Salud',
}


def clasificar_dimension(nombre_feature):
    if nombre_feature.startswith('V501'):
        return 'Sociodemografico'
    return DIMENSION_VARIABLE.get(nombre_feature, 'Sin clasificar')

print(f"Buscando archivos en: {RUTA_TRAIN.parent}")

print("\n" + "=" * 60)
print("TESIS: PREDICCION DE PERDIDA GESTACIONAL")
print("FASE 4: EXPLICABILIDAD GLOBAL (TreeSHAP)")
print("=" * 60)

# ==========================================
# PASO 3: CARGAR DATOS Y ENTRENAR EL MODELO GANADOR
# ==========================================
# Random Forest se usa como "modelo optimo" porque fue el mas consistente
# en los benchmarks de Modelos/ (mejor o segundo mejor F1/AUC en todas las
# corridas del proyecto). Mismos hiperparametros que en Modelos/
# entrenamiento_modelos.py, para que el modelo explicado aqui sea
# comparable con el ya evaluado.

df_train = pd.read_csv(RUTA_TRAIN, encoding='utf-8-sig')
df_test = pd.read_csv(RUTA_TEST, encoding='utf-8-sig')

feature_names = [c for c in df_train.columns if c != 'target']
X_train = df_train[feature_names]
y_train = df_train['target'].astype(int)
X_test = df_test[feature_names]
y_test = df_test['target'].astype(int)

print(f"\nTrain (balanceado): {X_train.shape[0]:,} filas")
print(f"Test (proporcion real, se explica sobre este): {X_test.shape[0]:,} filas")
print(f"Features: {feature_names}")

# One-Hot de V501 (fix C4), igual que en Modelos/entrenamiento_modelos_optimizado.py:
# se explica el mismo Random Forest que realmente se entrena y evalua ahi,
# no una version con el codigo numerico crudo de V501.
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
print(f"V501 codificado con One-Hot ({len(nombres_dummies_v501)} columnas): {nombres_dummies_v501}")

# Hiperparametros OPTIMOS encontrados en Modelos/entrenamiento_modelos_optimizado.py
# (busqueda sobre el set de validacion, tras el fix C3), no los valores por
# defecto: se explica el modelo realmente afinado, no una configuracion generica.
modelo = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=1, random_state=RANDOM_STATE, n_jobs=-1)
modelo.fit(X_train, y_train)
print("\nModelo (Random Forest, hiperparametros optimos) entrenado sobre el train balanceado")

# ==========================================
# PASO 4: CALCULAR VALORES SHAP (TreeExplainer)
# ==========================================

print("\n" + "-" * 60)
print("CALCULANDO VALORES SHAP (TreeExplainer)")
print("-" * 60)

explainer = shap.TreeExplainer(modelo)
explicacion = explainer(X_test)

# Para clasificadores binarios, shap puede devolver un array con una
# dimension extra por clase (n_filas, n_features, n_clases). Nos quedamos
# con la clase 1 (target=1, perdida gestacional), que es la que interesa
# explicar.
valores_shap = explicacion.values
if valores_shap.ndim == 3:
    valores_shap = valores_shap[:, :, 1]
    print("Se detecto salida por clase (binaria); se usa la clase 1 (perdida gestacional)")

print(f"Valores SHAP calculados: {valores_shap.shape[0]:,} filas x {valores_shap.shape[1]} features")

# ==========================================
# PASO 5: IMPORTANCIA GLOBAL = mean(|SHAP|)
# ==========================================
# Esta es la definicion estandar de "importancia global" en SHAP: el
# promedio del valor absoluto de la contribucion de cada variable a lo
# largo de todas las filas del test set. A diferencia de la importancia de
# Random Forest (que solo mide "cuanto reduce la impureza"), esta
# importancia esta directamente conectada a la magnitud del efecto en la
# prediccion final (la probabilidad de perdida gestacional).

mean_abs_shap = np.abs(valores_shap).mean(axis=0)
direccion_promedio = valores_shap.mean(axis=0)  # signo: si en promedio empuja el riesgo hacia arriba o abajo

resultados_shap = pd.DataFrame({
    'Feature': feature_names,
    'Mean_Abs_SHAP': mean_abs_shap,
    'Direccion_Promedio': direccion_promedio,
    'Dimension': [clasificar_dimension(f) for f in feature_names],
})
resultados_shap = resultados_shap.sort_values('Mean_Abs_SHAP', ascending=False).reset_index(drop=True)
resultados_shap.insert(0, 'Posicion', range(1, len(resultados_shap) + 1))
resultados_shap['Importancia_pct'] = (resultados_shap['Mean_Abs_SHAP'] / resultados_shap['Mean_Abs_SHAP'].sum() * 100).round(1)

print("\nRanking de importancia global (mean|SHAP|):")
print(resultados_shap.to_string(index=False))

resultados_shap.to_csv(RUTA_GUARDAR, index=False, encoding='utf-8-sig')
print(f"\nTabla guardada en: {RUTA_GUARDAR}")

# ==========================================
# PASO 6: JERARQUIA POR DIMENSION (Salud vs. Sociodemografico)
# ==========================================
# Indicador explicito de la Fase 4 en la matriz de consistencia: "Jerarquia
# de impacto (Factores de Salud vs. Factores Sociodemograficos)".

print("\n" + "-" * 60)
print("JERARQUIA POR DIMENSION: SALUD vs. SOCIODEMOGRAFICO")
print("-" * 60)

por_dimension = resultados_shap.groupby('Dimension')['Mean_Abs_SHAP'].sum().sort_values(ascending=False)
por_dimension_pct = (por_dimension / por_dimension.sum() * 100).round(1)

for dim in por_dimension.index:
    print(f"   {dim}: {por_dimension_pct[dim]}% de la contribucion total (SHAP)")

df_dimension = pd.DataFrame({
    'Dimension': por_dimension.index,
    'Suma_Mean_Abs_SHAP': por_dimension.values,
    'Porcentaje': por_dimension_pct.values,
})
df_dimension.to_csv(RUTA_GUARDAR_DIMENSION, index=False, encoding='utf-8-sig')
print(f"\nTabla guardada en: {RUTA_GUARDAR_DIMENSION}")

# ==========================================
# PASO 7: GRAFICOS (Beeswarm y barras de importancia)
# ==========================================

print("\n" + "-" * 60)
print("GENERANDO GRAFICOS")
print("-" * 60)

# Beeswarm: cada punto es una mujer del test set; el eje X es el valor
# SHAP (impacto en la prediccion), y el color representa si el valor de la
# variable para esa mujer fue alto (rojo) o bajo (azul). Es el grafico
# estandar para ver de un vistazo la importancia Y la direccion del efecto.
plt.figure(figsize=(9, 6))
explicacion_clase1 = shap.Explanation(
    values=valores_shap,
    base_values=explicacion.base_values[:, 1] if np.array(explicacion.base_values).ndim > 1 else explicacion.base_values,
    data=X_test.values,
    feature_names=feature_names,
)
shap.summary_plot(explicacion_clase1, X_test, show=False)
plt.title('SHAP Beeswarm - Explicabilidad Global (clase 1: perdida gestacional)')
plt.tight_layout()
plt.savefig(RUTA_BEESWARM, dpi=150, bbox_inches='tight')
plt.close()
print(f"Beeswarm guardado en: {RUTA_BEESWARM}")

# Barras: mean(|SHAP|) por variable, coloreado por dimension
plt.figure(figsize=(9, 6))
orden_grafico = resultados_shap.iloc[::-1]
colores = ['seagreen' if d == 'Salud' else 'steelblue' for d in orden_grafico['Dimension']]
plt.barh(orden_grafico['Feature'], orden_grafico['Mean_Abs_SHAP'], color=colores)
plt.xlabel('mean(|SHAP value|) - impacto promedio en la prediccion')
plt.title('Importancia global SHAP, por variable\n(verde = Salud, azul = Sociodemografico)')
plt.tight_layout()
plt.savefig(RUTA_BARRAS, dpi=150)
plt.close()
print(f"Grafico de barras guardado en: {RUTA_BARRAS}")

print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
print(f"\nVariable con mayor impacto global: {resultados_shap.iloc[0]['Feature']} "
      f"({resultados_shap.iloc[0]['Importancia_pct']}% de la contribucion total)")
print(f"Dimension dominante: {por_dimension_pct.idxmax()} ({por_dimension_pct.max()}%)")

print("\nEXPLICABILIDAD GLOBAL (SHAP) COMPLETADA")
print("=" * 60)
