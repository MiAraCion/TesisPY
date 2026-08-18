# ==========================================
# TESIS: PREDICCION DE PERDIDA GESTACIONAL
# BORUTA - SELECCION DE CARACTERISTICAS
# EJECUCION LOCAL
# ==========================================
#
# Este script toma el dataset consolidado (2022-2025), construido con los
# app.py de cada carpeta de anio y consolidar_excel.py en la raiz del
# proyecto, y aplica Boruta para confirmar cuales de las variables de la
# matriz de consistencia son estadisticamente relevantes para predecir la
# perdida gestacional.
#
# Boruta no reemplaza la seleccion teorica de variables (la matriz de
# consistencia): se aplica DESPUES, como una validacion adicional sobre el
# conjunto de variables que ya se justificaron por literatura/teoria.
#
# Dependencias: si falta alguna, instalar una sola vez desde la terminal
# (no dentro del script) con:
#   py -m pip install boruta pandas scikit-learn matplotlib

# ==========================================
# 1. IMPORTAR LIBRERIAS
# ==========================================

from pathlib import Path

import numpy as np
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

# ==========================================
# 2. CONFIGURACION DE RUTAS
# ==========================================
# Este script vive en boruta/, un nivel por debajo de la raiz del proyecto,
# asi que BASE_DIR sube un nivel para llegar a la raiz donde esta
# endes_2022_2025.csv. Usar Path(__file__) en vez de una ruta fija hace que
# el script funcione sin importar desde donde lo ejecutes.

BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_ARCHIVO = BASE_DIR / "endes_2022_2025.csv"
RUTA_GUARDAR = BASE_DIR / "boruta" / "resultados_boruta.csv"
RUTA_DATASET_BORUTA = BASE_DIR / "boruta" / "dataset_boruta_seleccionado.csv"
RUTA_DATASET_TOP12 = BASE_DIR / "boruta" / "dataset_top12_importancia.csv"

# Cuantas variables (por importancia, de mayor a menor) entran al dataset
# "top N". Se eligio 12 porque ahi hay una caida mas clara en el aporte
# marginal de cada variable siguiente (ver PASO 16): las primeras 12
# acumulan 85.9% de la importancia total, y la variable 13 en adelante
# aporta cada vez menos.
TOP_N = 12

print(f"Buscando archivo en: {RUTA_ARCHIVO}")

# ==========================================
# 5. CARGAR DATOS
# ==========================================

print("\n" + "=" * 60)
print("TESIS: PREDICCION DE PERDIDA GESTACIONAL")
print("BORUTA - SELECCION DE CARACTERISTICAS")
print("=" * 60)

try:
    df = pd.read_csv(RUTA_ARCHIVO, encoding='utf-8-sig')
    print("Dataset cargado correctamente")
    print(f"Dimensiones: {df.shape[0]:,} filas, {df.shape[1]:,} columnas")
except FileNotFoundError:
    print(f"ERROR: No se encontro el archivo en {RUTA_ARCHIVO}")
    print("Verifica la ruta en la seccion de configuracion")
    raise

print("\nDistribucion de target:")
print(df['target'].value_counts())
print(f"   Clase 0 (Sin perdida): {(df['target'] == 0).sum():,}")
print(f"   Clase 1 (Perdida): {(df['target'] == 1).sum():,}")

# ==========================================
# 6. SELECCION DE VARIABLES (MATRIZ DE CONSISTENCIA + AMPLIACION)
# ==========================================
# Estas son las predictoras + target que salen de app.py (2022-2025):
# la matriz de consistencia original, mas la ampliacion a "factores
# sociodemograficos y de salud" (ya no solo habitos): biomarcadores
# medidos (RECH5), calidad de la atencion obstetrica (REC41) y etnicidad.
# CASEID y anio se dejan fuera: CASEID es solo llave de identificacion,
# anio es informativo (no es un factor de riesgo, es la ronda de encuesta).
#
# V234 y QS900/QS901 ya NO existen en el dataset: V234 se excluyo por fuga
# de datos (su nulidad delataba el target casi perfectamente) y QS900/901
# (peso/talla autoreportados) se reemplazaron por el IMC medido de RECH5.

VARIABLES_MODELO = [
    # --- Factores Sociodemograficos ---
    'V012',   # Edad actual
    'V106',   # Nivel educativo mas alto aprobado
    'V025',   # Area de residencia (urbano/rural)
    'V190',   # Indice de riqueza (quintil)
    'V040',   # Altitud del conglomerado (metros)
    'V501',   # Estado civil actual
    'lengua_materna_indigena',   # Etnicidad (V131): 1 = lengua nativa, 0 = castellano
    'combustible_contaminante',  # Combustible de cocina (V161): 1 = lenia/carbon/kerosene

    # --- Salud: uso de anticonceptivos ---
    'V302',   # Alguna vez uso un metodo anticonceptivo
    'V313',   # Uso actual por tipo de metodo (ninguno/tradicional/moderno)

    # --- Antecedentes obstetricos ---
    'V222',   # Intervalo desde el ultimo nacimiento (meses)
    'V201',   # Total de hijos nacidos (paridad)

    # --- Salud: tabaco ---
    'QS200',  # Fumo cigarrillos en los ultimos 12 meses
    'QS202',  # Fuma diariamente
    'QS205C', # Cantidad de cigarrillos que fuma al dia

    # --- Salud: alcohol ---
    'QS206',  # Ha consumido alguna vez bebidas alcoholicas
    'QS208',  # Consumio alcohol en los ultimos 12 meses
    'QS210',  # Consumio alcohol en los ultimos 30 dias

    # --- Salud: biomarcadores medidos (RECH5) ---
    'hemoglobina',  # Nivel de hemoglobina en g/dL (anemia)
    'IMC',          # Indice de masa corporal medido por personal capacitado

    # --- Salud: atencion obstetrica (REC41) ---
    'M14',                         # Numero de controles prenatales
    'control_prenatal_calidad',    # Cuantos de los 5 examenes basicos se hicieron (0-5)
    'atencion_calificada_parto',   # 1 = atendida por medico/obstetra/enfermera/tecnico

    # --- Variable objetivo ---
    'target',
]

columnas_existentes = [col for col in VARIABLES_MODELO if col in df.columns]
columnas_faltantes = [col for col in VARIABLES_MODELO if col not in df.columns]

print(f"\nColumnas encontradas: {len(columnas_existentes)}")
if columnas_faltantes:
    print(f"Columnas NO encontradas: {columnas_faltantes}")
    print("Estas columnas se omitiran del modelo")

df_modelo = df[columnas_existentes].copy()
print(f"Dataset para Boruta: {df_modelo.shape[0]:,} filas, {df_modelo.shape[1]:,} columnas")

# ==========================================
# 7. LIMPIEZA NUMERICA (coma decimal y espacios en blanco)
# ==========================================
# El CSV fuente tiene dos problemas heredados de ENDES:
#   1) QS900 y QS901 (peso/talla) vienen con COMA decimal ("65,9" en vez
#      de "65.9"), asi que pandas los lee como texto, no como numero.
#   2) Varios campos vienen con un espacio en blanco (" ") en vez de un
#      valor vacio real, y pandas no lo reconoce como NaN por defecto.
#
# Esta funcion corrige ambos problemas para cualquier columna que deberia
# ser numerica, sin necesidad de listar caso por caso cual tiene cada
# problema.

def limpiar_numerico(serie):
    if serie.dtype == object:
        serie = serie.astype(str).str.strip()
        serie = serie.replace('', np.nan)
        serie = serie.str.replace(',', '.', regex=False)
    return pd.to_numeric(serie, errors='coerce')


print("\n" + "-" * 60)
print("LIMPIEZA DE FORMATO NUMERICO")
print("-" * 60)

for col in df_modelo.columns:
    antes_tipo = df_modelo[col].dtype
    df_modelo[col] = limpiar_numerico(df_modelo[col])
    if antes_tipo == object:
        print(f"   {col}: convertida de texto a numero")

# NOTA: el IMC ya no se construye aqui. app.py (2022-2025) ahora lo calcula
# a partir de HA40 de RECH5 (medido por personal capacitado durante la
# entrevista biomedica), en vez del peso/talla autoreportado en CSALUD01.
# La columna 'IMC' llega lista en endes_2022_2025.csv.

# ==========================================
# 9. MANEJO DE VALORES NULOS
# ==========================================
# Nota: parte de estos nulos son "no aplica" por patrones de salto del
# cuestionario (por ejemplo V222 queda vacio si la mujer no tiene un
# nacimiento previo, o QS202/QS205C quedan vacios si nunca fumo). Para
# esta corrida se imputan con mediana/moda por simplicidad; si se quiere
# distinguir "no aplica" de "no sabe" se recomienda revisar variable por
# variable contra el diccionario de ENDES antes de la version final.

print("\n" + "-" * 60)
print("MANEJO DE VALORES NULOS")
print("-" * 60)

for col in df_modelo.columns:
    nulos = df_modelo[col].isna().sum()
    if nulos > 0:
        print(f"   {col}: {nulos:,} ({nulos / len(df_modelo) * 100:.1f}%)")

for col in df_modelo.columns:
    if col == 'target':
        continue
    if df_modelo[col].dtype in ['int64', 'float64']:
        df_modelo[col] = df_modelo[col].fillna(df_modelo[col].median())
    else:
        df_modelo[col] = df_modelo[col].fillna(df_modelo[col].mode()[0])

# Las filas sin target no sirven ni para entrenar ni para validar
df_modelo = df_modelo.dropna(subset=['target']).copy()

print("\nValores nulos imputados")

# ==========================================
# 10. SEPARAR X (FEATURES) E y (TARGET)
# ==========================================

X = df_modelo.drop('target', axis=1).values
y = df_modelo['target'].values
feature_names = df_modelo.drop('target', axis=1).columns.tolist()

print(f"\nFeatures (X): {X.shape}")
print(f"Target (y): {y.shape}")
print(f"Nombres de features: {feature_names}")

# ==========================================
# 11. DIVIDIR EN TRAIN Y TEST
# ==========================================
# Boruta se ejecuta SOLO sobre el train set. Si se ejecutara sobre todo el
# dataset, la validacion posterior (paso 14) quedaria contaminada porque
# las variables se habrian elegido viendo datos que despues se usan para
# medir el desempenio.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=2023, stratify=y
)

print(f"\nTrain: {X_train.shape}")
print(f"Test: {X_test.shape}")

# ==========================================
# 12. BORUTA - SELECCION DE CARACTERISTICAS
# ==========================================
# No se estandariza (StandardScaler) antes de Boruta: Random Forest divide
# el espacio por umbrales de cada variable, no por distancias, asi que la
# escala de las variables no afecta el resultado. Estandarizar aqui no
# rompe nada, pero tampoco aporta y agrega un paso de mas.
#
# class_weight='balanced' es importante: el target esta desbalanceado
# (~91% clase 0 / ~9% clase 1). Sin esto, el Random Forest tiende a
# aprender patrones que solo explican bien a la clase mayoritaria, y la
# seleccion de variables queda sesgada hacia esos patrones.
#
# Eficiencia: n_jobs=-1 usa todos los nucleos disponibles para entrenar
# cada arbol del Random Forest en paralelo. early_stopping=True hace que
# Boruta corte antes de llegar a max_iter si el veredicto (confirmada o
# rechazada) de cada variable ya no cambia durante n_iter_no_change
# iteraciones seguidas, en vez de forzar siempre las 50 completas. Con
# ~29,000 filas y menos de 20 variables esto deberia tardar pocos minutos.

print("\n" + "=" * 60)
print("BORUTA - SELECCION DE CARACTERISTICAS")
print("=" * 60)

print("\nIniciando Boruta...")

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

print("\nEjecutando Boruta...")
boruta.fit(X_train, y_train)

# ==========================================
# 13. RESULTADOS DE BORUTA
# ==========================================
# BorutaPy no expone un atributo feature_importances_ propio (solo
# support_ y ranking_). Para obtener importancias interpretables y
# alineadas 1 a 1 con feature_names, se reentrena un Random Forest limpio
# sobre las mismas features de train (sin las variables shadow internas
# de Boruta) y se usan sus importancias solo para el grafico.

print("\n" + "-" * 60)
print("RESULTADOS DE BORUTA")
print("-" * 60)

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

resultados_boruta = pd.DataFrame({
    'Feature': feature_names,
    'Seleccionada': selected,
    'Ranking': ranking,
    'Importancia': rf_importancias.feature_importances_,
})

# Orden de mayor a menor puntaje (Importancia), no por Ranking de Boruta.
# Ranking y Importancia suelen coincidir pero no son lo mismo: Ranking sale
# de las comparaciones contra shadow features de Boruta, Importancia sale
# directo del Random Forest. Se ordena por Importancia porque es la
# puntuacion real, continua, que permite ademas calcular cuanto aporta
# cada variable de forma acumulada (ver Importancia_acumulada_pct).
resultados_boruta = resultados_boruta.sort_values('Importancia', ascending=False).reset_index(drop=True)
resultados_boruta.insert(0, 'Posicion', range(1, len(resultados_boruta) + 1))
resultados_boruta['Importancia_acumulada_pct'] = (
    resultados_boruta['Importancia'].cumsum() / resultados_boruta['Importancia'].sum() * 100
).round(1)

print(f"\nResumen de Boruta:")
print(f"   Total de features evaluadas: {len(feature_names)}")
print(f"   Features confirmadas como importantes: {selected.sum()}")
print(f"   Features rechazadas: {(~selected).sum()}")

print(f"\nFeatures seleccionadas por Boruta:")
for _, row in resultados_boruta[resultados_boruta['Seleccionada']].iterrows():
    print(f"   {row['Feature']} (Ranking: {row['Ranking']})")

print(f"\nFeatures rechazadas por Boruta:")
for _, row in resultados_boruta[~resultados_boruta['Seleccionada']].iterrows():
    print(f"   {row['Feature']} (Ranking: {row['Ranking']})")

# ==========================================
# 14. VISUALIZACION DE RESULTADOS
# ==========================================

plt.figure(figsize=(10, 8))
colores = ['green' if s else 'red' for s in resultados_boruta['Seleccionada']]
plt.barh(resultados_boruta['Feature'], resultados_boruta['Importancia'], color=colores)
plt.xlabel('Importancia (Random Forest)')
plt.title('Importancia de Features segun Boruta\n(Verde = Seleccionada, Rojo = Rechazada)')
plt.tight_layout()
plt.show()

# ==========================================
# 15. VALIDACION DE BORUTA
# ==========================================
# Con un target desbalanceado (~91/9), accuracy por si sola es enganosa:
# un modelo que siempre prediga "0" ya obtiene ~91% de accuracy sin haber
# aprendido nada util. Por eso se agregan recall, precision y F1 de la
# clase 1 (perdida gestacional), que es la clase que realmente interesa
# detectar, ademas de balanced_accuracy.

print("\n" + "-" * 60)
print("VALIDACION DE BORUTA")
print("-" * 60)


def evaluar(nombre, y_true, y_pred):
    print(f"\n{nombre}")
    print(f"   Accuracy:          {accuracy_score(y_true, y_pred):.4f}")
    print(f"   Balanced accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")
    print(f"   Precision (clase 1): {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"   Recall (clase 1):    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"   F1 (clase 1):        {f1_score(y_true, y_pred, zero_division=0):.4f}")


rf_all = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=2023)
rf_all.fit(X_train, y_train)
pred_all = rf_all.predict(X_test)
evaluar(f"Todas las features ({X_train.shape[1]})", y_test, pred_all)

X_train_boruta = X_train[:, selected]
X_test_boruta = X_test[:, selected]

rf_sel = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=2023)
rf_sel.fit(X_train_boruta, y_train)
pred_sel = rf_sel.predict(X_test_boruta)
evaluar(f"Features seleccionadas por Boruta ({selected.sum()})", y_test, pred_sel)

# ==========================================
# 16. CREAR NUEVO DATASET CON FEATURES SELECCIONADAS
# ==========================================

print("\n" + "-" * 60)
print("CREANDO NUEVO DATASET CON FEATURES SELECCIONADAS")
print("-" * 60)

features_seleccionadas = resultados_boruta[resultados_boruta['Seleccionada']]['Feature'].tolist()
columnas_finales = features_seleccionadas + ['target']
df_final_boruta = df_modelo[columnas_finales].copy()

print(f"Nuevo dataset (solo confirmadas por Boruta):")
print(f"   Filas: {df_final_boruta.shape[0]:,}")
print(f"   Columnas: {df_final_boruta.shape[1]:,}")
print(f"   Features seleccionadas: {len(features_seleccionadas)}")

print(f"\nColumnas finales:")
for col in df_final_boruta.columns:
    print(f"   - {col}")

# --- Dataset alternativo: top N por importancia acumulada (no solo las ---
# --- "Confirmadas" de Boruta, que aqui son solo 3) ---
features_top_n = resultados_boruta.head(TOP_N)['Feature'].tolist()
columnas_top_n = features_top_n + ['target']
df_top_n = df_modelo[columnas_top_n].copy()
importancia_acumulada_top_n = resultados_boruta.iloc[TOP_N - 1]['Importancia_acumulada_pct']

print(f"\nNuevo dataset (top {TOP_N} por importancia, {importancia_acumulada_top_n}% acumulado):")
print(f"   Filas: {df_top_n.shape[0]:,}")
print(f"   Columnas: {df_top_n.shape[1]:,}")
print(f"\nColumnas finales:")
for col in df_top_n.columns:
    print(f"   - {col}")

# ==========================================
# 17. GUARDAR RESULTADOS
# ==========================================

try:
    resultados_boruta.to_csv(RUTA_GUARDAR, index=False, encoding='utf-8-sig')
    print(f"\nResultados de Boruta guardados en: {RUTA_GUARDAR}")
except Exception as e:
    print(f"No se pudo guardar el archivo: {e}")

try:
    df_final_boruta.to_csv(RUTA_DATASET_BORUTA, index=False, encoding='utf-8-sig')
    print(f"Dataset con features seleccionadas (Boruta puro) guardado en: {RUTA_DATASET_BORUTA}")
except Exception as e:
    print(f"No se pudo guardar el dataset: {e}")

try:
    df_top_n.to_csv(RUTA_DATASET_TOP12, index=False, encoding='utf-8-sig')
    print(f"Dataset con top {TOP_N} por importancia guardado en: {RUTA_DATASET_TOP12}")
except Exception as e:
    print(f"No se pudo guardar el dataset top {TOP_N}: {e}")

# ==========================================
# 18. RESUMEN FINAL
# ==========================================

print("\n" + "=" * 60)
print("RESUMEN FINAL - BORUTA")
print("=" * 60)

print(f"\nDataset original para Boruta:")
print(f"   Filas: {df_modelo.shape[0]:,}")
print(f"   Columnas: {df_modelo.shape[1]:,}")

print(f"\nResultados de Boruta:")
print(f"   Features evaluadas: {len(feature_names)}")
print(f"   Features confirmadas como importantes: {selected.sum()}")
print(f"   Features rechazadas: {(~selected).sum()}")

print(f"\nDataset final (con features seleccionadas):")
print(f"   Filas: {df_final_boruta.shape[0]:,}")
print(f"   Columnas: {df_final_boruta.shape[1]:,}")

print(f"\nFeatures seleccionadas:")
for col in features_seleccionadas:
    print(f"   {col}")

print(f"\nFeatures rechazadas:")
rechazadas = resultados_boruta[~resultados_boruta['Seleccionada']]['Feature'].tolist()
for col in rechazadas:
    print(f"   {col}")

print(f"\nArchivos generados:")
print(f"   1. Resultados Boruta: {RUTA_GUARDAR}")
print(f"   2. Dataset final: {RUTA_DATASET_BORUTA}")

print("\nBORUTA COMPLETADO")
print("=" * 60)