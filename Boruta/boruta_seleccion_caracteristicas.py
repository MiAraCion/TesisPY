# ==========================================
# TESIS: PREDICCION DE PERDIDA GESTACIONAL
# BORUTA - SELECCION DE CARACTERISTICAS
# EJECUCION LOCAL
# ==========================================
#
# QUE HACE ESTE SCRIPT, EN UNA FRASE:
# toma el dataset consolidado (endes_2022_2025.csv, en la raiz del proyecto)
# y usa el algoritmo Boruta para decidir, con evidencia estadistica, cuales
# de las variables candidatas son realmente relevantes para predecir la
# perdida gestacional (columna 'target').
#
# COMO FUNCIONA BORUTA (idea general, antes de leer el codigo):
# 1) Por cada variable real (ej. V012 = edad), Boruta crea una copia
#    "sombra" (shadow feature) con los mismos valores pero barajados al azar
#    entre las filas. Una variable sombra, por construccion, NO puede tener
#    relacion real con el target (se destruyo esa relacion al barajarla).
# 2) Entrena un Random Forest con las variables reales + todas las sombras
#    juntas, y mide la importancia que el bosque le da a cada una.
# 3) Si una variable real tiene MAS importancia que la mejor de las sombras,
#    cuenta como un "acierto" (hit) en esa ronda.
# 4) Repite esto muchas iteraciones (rondas). Al final, aplica una prueba
#    estadistica sobre el numero de aciertos de cada variable: si acierta
#    consistentemente mas que el azar, se CONFIRMA como relevante; si nunca
#    supera a sus sombras, se RECHAZA.
# Esto es distinto de simplemente "ordenar por importancia": Boruta pregunta
# "¿esta variable es mejor que el ruido puro?", no solo "¿cual es la mas
# importante de las que tengo?".
#
# QUE NO HACE ESTE SCRIPT:
# no reemplaza la seleccion teorica de variables (la matriz de consistencia
# del proyecto). Se usa DESPUES, como evidencia estadistica adicional sobre
# variables que ya se justificaron por literatura.
#
# Dependencias (instalar una sola vez desde la terminal, no aqui adentro):
#   py -m pip install boruta pandas scikit-learn matplotlib

# ==========================================
# PASO 1: IMPORTAR LIBRERIAS
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
# PASO 2: CONFIGURACION DE RUTAS
# ==========================================
# Este script vive en boruta/, un nivel por debajo de la raiz del proyecto.
# BASE_DIR sube un nivel para llegar ahi, donde esta endes_2022_2025.csv.
# Usar Path(__file__) en vez de una ruta fija hace que el script funcione
# sin importar desde donde lo ejecutes (terminal, IDE, doble clic, etc.).

BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_ARCHIVO = BASE_DIR / "endes_2022_2025.csv"
RUTA_GUARDAR = BASE_DIR / "boruta" / "resultados_boruta.csv"
RUTA_DATASET_TOPN = BASE_DIR / "boruta" / "dataset_topN_importancia.csv"

# Cuantas variables (por importancia, de mayor a menor) entran al dataset
# "top N" que se guarda al final, ademas del dataset con solo las
# confirmadas por Boruta. Se puede ajustar despues de ver la tabla de
# importancia acumulada que imprime el PASO 8.
TOP_N = 10

print(f"Buscando archivo en: {RUTA_ARCHIVO}")

# ==========================================
# PASO 3: CARGAR EL DATASET CONSOLIDADO
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
    print("Verifica que exista endes_2022_2025.csv en la raiz del proyecto")
    raise

print("\nDistribucion de target:")
print(df['target'].value_counts())
print(f"   Clase 0 (Sin perdida): {(df['target'] == 0).sum():,}")
print(f"   Clase 1 (Perdida): {(df['target'] == 1).sum():,}")

# ==========================================
# PASO 4: SELECCION DE VARIABLES CANDIDATAS
# ==========================================
# Estas son las predictoras que produce app.py (2022-2025) para esta rama
# del proyecto. CASEID y anio se dejan fuera (CASEID es solo llave de
# identificacion, anio es la ronda de encuesta, no un factor de riesgo).
#
# V234 (otras perdidas antes de la ultima) se EXCLUYE A PROPOSITO, aunque
# sigue estando en endes_2022_2025.csv. Se valido con una tabla cruzada en
# una sesion anterior de este proyecto: ENDES solo pregunta V234 cuando
# V228 == 1, y dentro de este dataset V228 == 1 equivale casi exactamente a
# target == 1 (los casos de perdida tardia con V228 == 1 ya se descartaron
# al construir el target en app.py). O sea que si V234 tiene o no tiene
# respuesta ya delata el target, sin importar el contenido de la respuesta.
# Es fuga de datos (data leakage), no una variable predictora real. Si se
# incluyera, Boruta la "confirmaria" con altisima importancia, pero seria
# un resultado invalido.

VARIABLES_MODELO = [
    # --- Factores Sociodemograficos ---
    'V012',   # Edad actual
    'V106',   # Nivel educativo mas alto aprobado
    'V025',   # Area de residencia (urbano/rural)
    'V190',   # Indice de riqueza (quintil)
    'V040',   # Altitud del conglomerado (metros)
    'V501',   # Estado civil actual

    # --- Salud: uso de anticonceptivos ---
    'V302',   # Alguna vez uso un metodo anticonceptivo
    'V313',   # Uso actual por tipo de metodo (ninguno/tradicional/moderno)

    # --- Antecedentes obstetricos ---
    'V222',   # Intervalo entre el ultimo nacimiento y la entrevista (meses)
    'V201',   # Total de hijos nacidos (paridad)

    # --- Salud: tabaco ---
    'QS200',  # Fumo cigarrillos en los ultimos 12 meses
    'QS202',  # Fuma diariamente
    'QS205C', # Cantidad de cigarrillos que fuma al dia (muy poca cobertura,
              # se deja para que Boruta la evalue y la rechace con evidencia,
              # en vez de sacarla a mano)

    # --- Salud: alcohol ---
    'QS206',  # Ha consumido alguna vez bebidas alcoholicas
    'QS208',  # Consumio alcohol en los ultimos 12 meses
    'QS209',  # Consumio alcohol 12 o mas dias en los ultimos 12 meses (frecuencia)
    'QS210',  # Consumio alcohol en los ultimos 30 dias

    # --- Salud: IMC (se construye en el PASO 6 desde peso/talla) ---
    'QS900',  # Peso en kilogramos
    'QS901',  # Talla en centimetros

    # --- Salud: comorbilidad autoreportada ---
    # OJO: refleja el diagnostico ACTUAL (al momento de la entrevista), no
    # si la diabetes ya existia cuando ocurrio el embarazo/perdida evaluado.
    # ENDES no registra fecha de diagnostico. Ver nota en README.md.
    'QS109',  # Le diagnosticaron diabetes o azucar alta

    # --- Salud: controles prenatales ---
    'M14',    # Numero de controles prenatales del ultimo embarazo

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
# PASO 5: LIMPIEZA NUMERICA (defensiva)
# ==========================================
# endes_2022_2025.csv ya deberia venir con numeros limpios (se corrigio el
# problema de coma decimal en QS900/QS901 directamente en app.py), pero esta
# funcion se deja como red de seguridad: convierte a numero cualquier
# columna que llegue como texto, tratando comas decimales y espacios en
# blanco como si fueran punto decimal o vacio, respectivamente. Si todo
# viene limpio, esta funcion no cambia nada.

def limpiar_numerico(serie):
    if serie.dtype == object:
        serie = serie.astype(str).str.strip()
        serie = serie.replace('', np.nan)
        serie = serie.str.replace(',', '.', regex=False)
    return pd.to_numeric(serie, errors='coerce')


print("\n" + "-" * 60)
print("LIMPIEZA DE FORMATO NUMERICO (defensiva)")
print("-" * 60)

for col in df_modelo.columns:
    antes_tipo = df_modelo[col].dtype
    df_modelo[col] = limpiar_numerico(df_modelo[col])
    if antes_tipo == object:
        print(f"   {col}: convertida de texto a numero")

print("(sin conversiones = el archivo ya venia limpio)")

# ==========================================
# PASO 6: CONSTRUIR IMC A PARTIR DE PESO Y TALLA
# ==========================================
# La matriz de consistencia define el indicador como IMC (peso / talla^2),
# no como peso y talla por separado. Se construye la variable derivada y se
# descartan las dos columnas crudas para no duplicar la misma senal (peso y
# talla solas no dicen mucho sobre riesgo; su combinacion si).

print("\n" + "-" * 60)
print("CONSTRUYENDO IMC = peso_kg / (talla_m ** 2)")
print("-" * 60)

talla_metros = df_modelo['QS901'] / 100
df_modelo['IMC'] = df_modelo['QS900'] / (talla_metros ** 2)
df_modelo = df_modelo.drop(columns=['QS900', 'QS901'])

# Valores fuera de un rango fisiologicamente posible (ej. errores de tipeo
# o codigos sentinela mal interpretados como numero) se tratan como dato
# invalido, no como un IMC real.
fuera_rango = ~df_modelo['IMC'].between(10, 70) & df_modelo['IMC'].notna()
print(f"IMC calculado. Valores fuera de rango fisiologico (10-70): {fuera_rango.sum():,}")
df_modelo.loc[fuera_rango, 'IMC'] = np.nan
print(f"IMC valido en: {df_modelo['IMC'].notna().sum():,} de {len(df_modelo):,} filas")

# ==========================================
# PASO 7: MANEJO DE VALORES NULOS
# ==========================================
# Nota: parte de estos nulos son "no aplica" por patrones de salto del
# cuestionario (ej. V222 queda vacio si la mujer no tiene un nacimiento
# previo, QS205C queda vacio si no fuma diariamente). Para esta corrida se
# imputan con mediana (numericas) o moda (categoricas) por simplicidad. Es
# una limitacion documentada: no distingue "no aplica" de "no sabe".

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
# PASO 8: SEPARAR X (FEATURES) E y (TARGET)
# ==========================================

X = df_modelo.drop('target', axis=1).values
y = df_modelo['target'].values
feature_names = df_modelo.drop('target', axis=1).columns.tolist()

print(f"\nFeatures (X): {X.shape}")
print(f"Target (y): {y.shape}")
print(f"Nombres de features ({len(feature_names)}): {feature_names}")

# ==========================================
# PASO 9: DIVIDIR EN TRAIN Y TEST
# ==========================================
# Boruta se ejecuta SOLO sobre el train set. Si se ejecutara sobre todo el
# dataset, la validacion posterior (PASO 12) quedaria contaminada: las
# variables se habrian elegido viendo los mismos datos que despues se usan
# para medir que tan bien funciona esa eleccion. stratify=y mantiene la
# misma proporcion de clase 0/1 en train y test que en el dataset completo
# (importante porque el target esta desbalanceado, ~91% / 9%).

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=2023, stratify=y
)

print(f"\nTrain: {X_train.shape}")
print(f"Test: {X_test.shape}")

# ==========================================
# PASO 10: BORUTA - SELECCION DE CARACTERISTICAS
# ==========================================
# Parametros clave y por que estan asi:
#
# - class_weight='balanced' en el Random Forest: el target esta
#   desbalanceado (~91% clase 0 / ~9% clase 1). Sin esto, el bosque
#   aprende patrones que explican bien a la clase mayoritaria y practicamente
#   ignora la minoritaria, sesgando que variables parecen "importantes".
#
# - No se usa StandardScaler antes de Boruta: un Random Forest divide el
#   espacio por umbrales de cada variable (ej. "V012 > 30"), no por
#   distancias entre puntos, asi que la escala de las variables no afecta
#   el resultado. Estandarizar aqui no rompe nada, pero tampoco aporta.
#
# - n_jobs=-1: usa todos los nucleos del procesador disponibles para
#   entrenar los arboles del bosque en paralelo (mas rapido).
#
# - early_stopping=True, n_iter_no_change=15: Boruta corta las iteraciones
#   apenas el veredicto (confirmada/rechazada) de cada variable deja de
#   cambiar durante 15 rondas seguidas, en vez de forzar siempre las
#   max_iter=50 completas. Con este tamano de dataset (~29,000 filas, ~19
#   variables) el proceso completo tarda segundos, no minutos.

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
# PASO 11: TABLA DE RESULTADOS (ordenada de mayor a menor puntaje)
# ==========================================
# BorutaPy no expone un atributo feature_importances_ propio (solo
# support_ y ranking_). Para obtener importancias interpretables y
# alineadas 1 a 1 con feature_names, se reentrena un Random Forest limpio
# sobre las mismas features de train (sin las variables shadow internas de
# Boruta) y se usan sus importancias para ordenar la tabla y graficar.
#
# La tabla se ordena por 'Importancia' (puntaje real, continuo del Random
# Forest), no por 'Ranking' de Boruta (que sale de comparar contra shadow
# features). Suelen coincidir pero no son lo mismo. Ademas se calcula
# 'Importancia_acumulada_pct': que porcentaje del total explican las N
# variables con mejor puntaje hasta esa fila (sirve para justificar un
# corte "top N" con un numero real en vez de a ojo).

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

resultados_boruta = resultados_boruta.sort_values('Importancia', ascending=False).reset_index(drop=True)
resultados_boruta.insert(0, 'Posicion', range(1, len(resultados_boruta) + 1))
resultados_boruta['Importancia_acumulada_pct'] = (
    resultados_boruta['Importancia'].cumsum() / resultados_boruta['Importancia'].sum() * 100
).round(1)

print(f"\nResumen de Boruta:")
print(f"   Total de features evaluadas: {len(feature_names)}")
print(f"   Features confirmadas como importantes: {selected.sum()}")
print(f"   Features rechazadas: {(~selected).sum()}")

print(f"\nTabla completa (de mayor a menor importancia):")
print(resultados_boruta.to_string(index=False))

# ==========================================
# PASO 12: VISUALIZACION DE RESULTADOS
# ==========================================

plt.figure(figsize=(10, 8))
orden_grafico = resultados_boruta.iloc[::-1]  # invertido: la mas importante queda arriba en el grafico horizontal
colores = ['green' if s else 'red' for s in orden_grafico['Seleccionada']]
plt.barh(orden_grafico['Feature'], orden_grafico['Importancia'], color=colores)
plt.xlabel('Importancia (Random Forest)')
plt.title('Importancia de Features segun Boruta\n(Verde = Confirmada, Rojo = Rechazada)')
plt.tight_layout()
plt.savefig(BASE_DIR / "boruta" / "grafico_importancia.png", dpi=150)
print(f"\nGrafico guardado en: {BASE_DIR / 'boruta' / 'grafico_importancia.png'}")

# ==========================================
# PASO 13: VALIDACION - CON TODAS LAS FEATURES VS. SOLO LAS CONFIRMADAS
# ==========================================
# Con un target desbalanceado (~91/9), accuracy por si sola es enganosa: un
# modelo que siempre prediga "0" ya obtiene ~91% de accuracy sin haber
# aprendido nada util. Por eso se reportan tambien recall, precision y F1
# de la clase 1 (perdida gestacional, la que realmente interesa detectar),
# ademas de balanced_accuracy.

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
pred_all = rf_all.predict(X_test)
evaluar(f"Todas las features ({X_train.shape[1]})", y_test, pred_all)

X_train_boruta = X_train[:, selected]
X_test_boruta = X_test[:, selected]

rf_sel = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=2023)
rf_sel.fit(X_train_boruta, y_train)
pred_sel = rf_sel.predict(X_test_boruta)
evaluar(f"Solo features confirmadas por Boruta ({selected.sum()})", y_test, pred_sel)

# ==========================================
# PASO 14: CREAR DATASETS DE SALIDA
# ==========================================
# Se generan DOS datasets alternativos, porque Boruta suele ser muy
# conservador (a veces confirma muy pocas variables) y no siempre conviene
# restringirse solo a esas:
#
#   a) dataset con SOLO las confirmadas por Boruta (maxima evidencia
#      estadistica, minimo respaldo teorico si quedan pocas)
#   b) dataset con el TOP_N por importancia acumulada (mezcla evidencia
#      estadistica con margen para mantener variables con respaldo teorico
#      fuerte aunque Boruta no las haya confirmado individualmente)

print("\n" + "-" * 60)
print("CREANDO DATASETS DE SALIDA")
print("-" * 60)

features_confirmadas = resultados_boruta[resultados_boruta['Seleccionada']]['Feature'].tolist()
df_confirmadas = df_modelo[features_confirmadas + ['target']].copy()

print(f"\na) Dataset con solo las CONFIRMADAS por Boruta:")
print(f"   Filas: {df_confirmadas.shape[0]:,} | Columnas: {df_confirmadas.shape[1]} | Features: {len(features_confirmadas)}")
for col in df_confirmadas.columns:
    print(f"   - {col}")

features_top_n = resultados_boruta.head(TOP_N)['Feature'].tolist()
df_top_n = df_modelo[features_top_n + ['target']].copy()
importancia_acumulada_top_n = resultados_boruta.iloc[TOP_N - 1]['Importancia_acumulada_pct']

print(f"\nb) Dataset con el TOP {TOP_N} por importancia ({importancia_acumulada_top_n}% acumulado):")
print(f"   Filas: {df_top_n.shape[0]:,} | Columnas: {df_top_n.shape[1]}")
for col in df_top_n.columns:
    print(f"   - {col}")

# ==========================================
# PASO 15: GUARDAR RESULTADOS
# ==========================================

try:
    resultados_boruta.to_csv(RUTA_GUARDAR, index=False, encoding='utf-8-sig')
    print(f"\nTabla de resultados guardada en: {RUTA_GUARDAR}")
except Exception as e:
    print(f"No se pudo guardar la tabla de resultados: {e}")

try:
    df_top_n.to_csv(RUTA_DATASET_TOPN, index=False, encoding='utf-8-sig')
    print(f"Dataset top {TOP_N} guardado en: {RUTA_DATASET_TOPN}")
except Exception as e:
    print(f"No se pudo guardar el dataset top {TOP_N}: {e}")

# ==========================================
# PASO 16: RESUMEN FINAL
# ==========================================

print("\n" + "=" * 60)
print("RESUMEN FINAL - BORUTA")
print("=" * 60)

print(f"\nDataset original para Boruta:")
print(f"   Filas: {df_modelo.shape[0]:,}")
print(f"   Columnas evaluadas (sin target): {len(feature_names)}")

print(f"\nFeatures confirmadas por Boruta ({selected.sum()}):")
for col in features_confirmadas:
    print(f"   {col}")

print(f"\nFeatures rechazadas ({(~selected).sum()}):")
rechazadas = resultados_boruta[~resultados_boruta['Seleccionada']]['Feature'].tolist()
for col in rechazadas:
    print(f"   {col}")

print(f"\nArchivos generados en boruta/:")
print(f"   1. resultados_boruta.csv       (tabla completa ordenada por importancia)")
print(f"   2. dataset_topN_importancia.csv (top {TOP_N} variables + target)")
print(f"   3. grafico_importancia.png     (grafico de barras)")

print("\nBORUTA COMPLETADO")
print("=" * 60)
