import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# CONFIGURACION (especifica del anio 2025)
# ==========================================

ANIO = 2025
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "endes_filtrada_ml_2025.csv"

# Nombres de archivo tal como existen en las carpetas de este anio
ARCHIVO_1631 = "REC0111_2025.csv"
ARCHIVO_1632 = "RE223132_2025.csv"
ARCHIVO_1633 = "REC41_2025.csv"
ARCHIVO_1635 = "RE516171_2025.csv"
ARCHIVO_1638 = "RECH5_2025.csv"
ARCHIVO_1640 = "CSALUD01_2025.csv"

# Columnas finales del dataset, segun la tabla de la matriz de consistencia
# V234 se excluye a proposito: es fuga de datos (ver PASO 4 en la
# conversacion del proyecto - su nulidad delata el target casi perfectamente
# por el patron de salto de la encuesta).
COLUMNAS_FINALES = [
    'CASEID',
    # --- Factores Sociodemograficos (1631) ---
    'V012', 'V106', 'V025', 'V190', 'V040',
    'lengua_materna_indigena',    # derivada de V131
    'combustible_contaminante',   # derivada de V161
    # --- Factores Sociodemograficos (1635) ---
    'V501',
    # --- Salud: uso de anticonceptivos (1632) ---
    'V302', 'V313',
    # --- Antecedentes obstetricos (1632, mi preferencia) ---
    'V222', 'V201',
    # --- Salud: tabaco y alcohol (1640) ---
    'QS200', 'QS202', 'QS205C',
    'QS206', 'QS208', 'QS210',
    # --- Salud: biomarcadores medidos (1638) ---
    'hemoglobina',   # derivada de HA53 (g/dL), solo si la medicion fue valida
    'IMC',           # derivada de HA40 (medido por personal capacitado)
    # --- Salud: calidad de la atencion obstetrica (1633) ---
    'M14',                          # numero de controles prenatales
    'control_prenatal_calidad',     # derivada de M42A-E (0-5)
    'atencion_calificada_parto',    # derivada de M3A-D
    # --- Variable objetivo ---
    'target',
]


def detectar_separador(ruta_archivo):
    """Detecta si el CSV usa ',' o ';' (algunos archivos ENDES no usan el mismo separador)"""
    with open(ruta_archivo, 'r', encoding='latin-1') as f:
        primera_linea = f.readline()
    return ';' if primera_linea.count(';') > primera_linea.count(',') else ','


def cargar_csv(ruta_archivo):
    """Carga un CSV detectando separador y codificacion, normaliza la primera columna (ID1, viene con BOM)"""
    sep = detectar_separador(ruta_archivo)
    try:
        df = pd.read_csv(ruta_archivo, encoding='latin-1', low_memory=False, sep=sep)
    except UnicodeDecodeError:
        df = pd.read_csv(ruta_archivo, encoding='utf-8', low_memory=False, sep=sep)
    df = df.rename(columns={df.columns[0]: 'ID1'})
    return df


def verificar_anio(df, nombre_archivo):
    """Alerta si el archivo cargado no corresponde al anio esperado (evita el bug de anios cruzados)"""
    anios_encontrados = df['ID1'].unique().tolist()
    if anios_encontrados != [ANIO]:
        print(f"   ADVERTENCIA: {nombre_archivo} deberia ser del anio {ANIO} pero tiene ID1={anios_encontrados}")


# ==========================================
# PASO 1: CARGAR MODULO 1632 (RE223132) - Historia de embarazos
# ==========================================

print("=" * 70)
print(f"PASO 1: CARGANDO MODULO 1632 (RE223132) - Historia de embarazos [{ANIO}]")
print("=" * 70)

ruta_1632 = BASE_DIR / "1632" / ARCHIVO_1632
df_1632 = cargar_csv(ruta_1632)
verificar_anio(df_1632, ARCHIVO_1632)
print(f"Cargado: {df_1632.shape[0]:,} filas, {df_1632.shape[1]} columnas")
print(f"Anio de la encuesta (ID1): {df_1632['ID1'].unique().tolist()}")

# ==========================================
# PASO 2: CARGAR MODULO 1631 (REC0111) - Sociodemograficas + edad real
# ==========================================

print("\n" + "=" * 70)
print(f"PASO 2: CARGANDO MODULO 1631 (REC0111) - Sociodemograficas [{ANIO}]")
print("=" * 70)

ruta_1631 = BASE_DIR / "1631" / ARCHIVO_1631
df_1631 = cargar_csv(ruta_1631)
verificar_anio(df_1631, ARCHIVO_1631)
print(f"Cargado: {df_1631.shape[0]:,} filas, {df_1631.shape[1]} columnas")

# HHID + V003 se guardan solo para poder enlazar mas adelante con CSALUD01 (1640)
# y RECH5 (1638), que no tienen CASEID sino HHID + numero de linea de la
# persona dentro del hogar. V131 (etnicidad) y V161 (combustible de cocina)
# se traen ahora para construir variables derivadas mas adelante.
df_1631 = df_1631[['CASEID', 'HHID', 'V003', 'V012', 'V106', 'V025', 'V190', 'V040', 'V131', 'V161']]

# ==========================================
# PASO 3: INNER JOIN 1632 + 1631 (para tener V012, edad real, antes de filtrar)
# ==========================================

print("\n" + "=" * 70)
print("PASO 3: INNER JOIN 1632 + 1631 (para disponer de V012 antes de filtrar)")
print("=" * 70)

df = df_1632.merge(df_1631, on='CASEID', how='inner')
print(f"CASEID: {df_1632['CASEID'].nunique():,} -> {df['CASEID'].nunique():,}")
print(f"Dimensiones: {df.shape}")

# ==========================================
# PASO 4: APLICAR FILTROS Y CONSTRUIR TARGET
# ==========================================

print("\n" + "=" * 70)
print("PASO 4: APLICANDO FILTROS")
print("=" * 70)

registros_iniciales = len(df)
print(f"\nRegistros iniciales: {registros_iniciales:,}")

cols_clave = ['V012', 'V201', 'V213', 'V228', 'V233']
for col in cols_clave:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# --- FILTRO 1: EDAD DE 15 A 49 ANIOS (V012 = edad ACTUAL) ---
print("\nFILTRO 1: Edad de 15 a 49 anios (V012 - edad actual de la entrevistada)")
f1_condicion = (df['V012'] >= 15) & (df['V012'] <= 49)
excluidos_f1 = len(df[~f1_condicion])
df = df[f1_condicion].copy()
print(f"   Excluidos: {excluidos_f1:,} | Quedan: {len(df):,}")

# --- FILTRO 2: AL MENOS 1 GESTACION (V201 > 0 o V228 == 1) ---
print("\nFILTRO 2: Al menos 1 gestacion (V201 > 0 o V228 == 1)")
f2_condicion = (df['V201'] > 0) | (df['V228'] == 1)
excluidos_f2 = len(df[~f2_condicion])
df = df[f2_condicion].copy()
print(f"   Excluidos: {excluidos_f2:,} | Quedan: {len(df):,}")

# --- FILTRO 3: EXCLUIR EMBARAZADAS ACTUALES (V213 == 1) ---
print("\nFILTRO 3: Excluir embarazadas actuales (V213 == 1)")
f3_condicion = df['V213'] != 1
excluidos_f3 = len(df[~f3_condicion])
df = df[f3_condicion].copy()
print(f"   Excluidos: {excluidos_f3:,} | Quedan: {len(df):,}")

# --- FILTRO 4: ELIMINAR PERDIDAS CON DURACION INCONSISTENTE ---
print("\nFILTRO 4: Eliminar perdidas con duracion inconsistente (V233)")
f4_inconsistentes = (df['V228'] == 1) & (df['V233'].isna() | df['V233'].isin([97, 98]))
excluidos_f4 = len(df[f4_inconsistentes])
df = df[~f4_inconsistentes].copy()
print(f"   Excluidos: {excluidos_f4:,} | Quedan: {len(df):,}")

# --- FILTRO 5: CREAR TARGET (VARIABLE OBJETIVO) ---
print("\nFILTRO 5: Crear variable objetivo 'target'")
condiciones = [
    (df['V228'] == 1) & (df['V233'].isin([1, 2, 3, 4, 5])),  # Perdida < 6 meses
    (df['V228'] == 0),  # Sin perdida
]
df['target'] = np.select(condiciones, [1, 0], default=np.nan)
excluidos_f5 = int(df['target'].isna().sum())
df = df.dropna(subset=['target']).copy()
df['target'] = df['target'].astype(int)
print(f"   Excluidos (perdidas tardias/inconsistentes): {excluidos_f5:,} | Quedan: {len(df):,}")

# --- RESUMEN DE FILTROS ---
print("\n" + "=" * 70)
print("RESUMEN DE FILTROS APLICADOS")
print("=" * 70)
print(f"Total inicial: {registros_iniciales:,}")
print(f"Excluidos (edad): {excluidos_f1:,}")
print(f"Excluidos (sin gestacion): {excluidos_f2:,}")
print(f"Excluidos (embarazadas): {excluidos_f3:,}")
print(f"Excluidos (duracion inconsistente): {excluidos_f4:,}")
print(f"Excluidos (perdidas tardias): {excluidos_f5:,}")
print(f"Quedan (antes de cruzar con los demas modulos): {len(df):,}")

caseids_filtrados = df['CASEID'].unique().tolist()

# ==========================================
# PASO 5: CRUZAR CON LOS DEMAS MODULOS (INNER JOIN)
# ==========================================

print("\n" + "=" * 70)
print("PASO 5: CRUZANDO CON LOS DEMAS MODULOS (INNER JOIN)")
print("=" * 70)

# --- 1635: Estado civil ---
print("\nModulo 1635 (RE516171) - Estado civil")
ruta_1635 = BASE_DIR / "1635" / ARCHIVO_1635
df_1635 = cargar_csv(ruta_1635)
verificar_anio(df_1635, ARCHIVO_1635)
df_1635 = df_1635[df_1635['CASEID'].isin(caseids_filtrados)][['CASEID', 'V501']]

antes = df['CASEID'].nunique()
df = df.merge(df_1635, on='CASEID', how='inner')
print(f"   CASEID: {antes:,} -> {df['CASEID'].nunique():,}")

# --- 1633: Controles prenatales, calidad del control y atencion del parto ---
# REC41 tiene una fila POR EMBARAZO (indice MIDX), no por mujer.
# Nos quedamos con el embarazo mas reciente (MIDX == 1) para tener 1 fila por CASEID.
# Se valido con una tabla cruzada que M3A y M42A NO tienen el problema de
# V234: su nulidad esta repartida ~91%/9% entre target 0/1, igual que la
# proporcion base del dataset, o sea que no delatan el target.
print("\nModulo 1633 (REC41) - Controles prenatales, calidad y atencion del parto")
ruta_rec41 = BASE_DIR / "1633" / ARCHIVO_1633
df_rec41 = cargar_csv(ruta_rec41)
verificar_anio(df_rec41, ARCHIVO_1633)

cols_rec41 = ['CASEID', 'M14', 'M3A', 'M3B', 'M3C', 'M3D', 'M42A', 'M42B', 'M42C', 'M42D', 'M42E']
cols_rec41 = [c for c in cols_rec41 if c in df_rec41.columns]
df_rec41 = df_rec41[df_rec41['MIDX'] == 1][cols_rec41]
df_rec41 = df_rec41[df_rec41['CASEID'].isin(caseids_filtrados)]

antes = df['CASEID'].nunique()
df = df.merge(df_rec41, on='CASEID', how='inner')
print(f"   CASEID: {antes:,} -> {df['CASEID'].nunique():,}")

# --- 1638: Biomarcadores medidos (hemoglobina, IMC oficial) ---
# RECH5 tampoco tiene CASEID: se enlaza por HHID + numero de orden en el
# hogar (HA0 == V003), mismo patron que CSALUD01. A diferencia de CSALUD01,
# este archivo cubre a TODAS las mujeres del cuestionario individual
# (verificado: 100% de cobertura), no solo una submuestra.
print("\nModulo 1638 (RECH5) - Biomarcadores (hemoglobina, IMC medido)")
ruta_rech5 = BASE_DIR / "1638" / ARCHIVO_1638
df_rech5 = cargar_csv(ruta_rech5)
verificar_anio(df_rech5, ARCHIVO_1638)

df_rech5 = df_rech5[['HHID', 'HA0', 'HA13', 'HA40', 'HA53', 'HA55']]
df_rech5 = df_rech5.rename(columns={'HA0': 'V003'})

antes = df['CASEID'].nunique()
df = df.merge(df_rech5, on=['HHID', 'V003'], how='inner')
print(f"   CASEID: {antes:,} -> {df['CASEID'].nunique():,}")

# --- 1640: Habitos de salud (tabaco, alcohol) ---
# CSALUD01 NO tiene CASEID (es un cuestionario de hogar, no individual).
# Se enlaza por HHID + numero de linea de la persona dentro del hogar (QSNUMERO == V003).
# QS900/QS901 (peso/talla autoreportados) ya NO se usan: el IMC ahora viene
# medido de RECH5 (mas preciso y con mejor cobertura).
print("\nModulo 1640 (CSALUD01) - Habitos de salud")
print("   (Enlazado por HHID + linea de persona, no por CASEID)")
ruta_csalud01 = BASE_DIR / "1640_cuestionario_del_hogar" / ARCHIVO_1640
df_csalud = cargar_csv(ruta_csalud01)
verificar_anio(df_csalud, ARCHIVO_1640)

cols_habitos = ['QS200', 'QS202', 'QS205C', 'QS206', 'QS208', 'QS210']
cols_habitos = [c for c in cols_habitos if c in df_csalud.columns]
df_csalud = df_csalud[['HHID', 'QSNUMERO'] + cols_habitos]
df_csalud = df_csalud.rename(columns={'QSNUMERO': 'V003'})

antes = df['CASEID'].nunique()
df = df.merge(df_csalud, on=['HHID', 'V003'], how='inner')
print(f"   CASEID: {antes:,} -> {df['CASEID'].nunique():,}")
print("   NOTA: CSALUD01 solo se aplica a una submuestra de hogares en ENDES,")
print("   por eso esta union reduce el N de forma esperada (no es un error).")

# ==========================================
# PASO 5.5: CONSTRUIR VARIABLES DERIVADAS
# ==========================================

print("\n" + "=" * 70)
print("PASO 5.5: CONSTRUYENDO VARIABLES DERIVADAS")
print("=" * 70)

# --- Etnicidad: lengua materna indigena (V131) ---
# Codigos 1-9 = lenguas nativas (quechua, aimara, etc.), 10 = castellano.
# 11-12 (lenguas extranjeras) quedan como NaN: no son indigenas peruanas
# ni corresponden a la categoria "castellano".
df['V131'] = pd.to_numeric(df['V131'], errors='coerce')
df['lengua_materna_indigena'] = np.where(
    df['V131'].between(1, 9), 1,
    np.where(df['V131'] == 10, 0, np.nan)
)
print(f"lengua_materna_indigena: {df['lengua_materna_indigena'].value_counts(dropna=False).to_dict()}")

# --- Combustible de cocina: contaminante vs limpio (V161) ---
# Limpio = electricidad/gas (1,2,3). Contaminante = kerosene/carbon/lenia/
# biomasa (5,6,7,8,9,10,11). Codigos 95 (no cocina) y 97 (no residente)
# quedan como NaN por no ser un tipo de combustible real.
df['V161'] = pd.to_numeric(df['V161'], errors='coerce')
df['combustible_contaminante'] = np.where(
    df['V161'].isin([1, 2, 3]), 0,
    np.where(df['V161'].isin([5, 6, 7, 8, 9, 10, 11]), 1, np.nan)
)
print(f"combustible_contaminante: {df['combustible_contaminante'].value_counts(dropna=False).to_dict()}")

# --- Hemoglobina (HA53) e IMC (HA40): solo validas si la medicion salio bien ---
# HA55 == 0 (hemoglobina) y HA13 == 0 (medicion general) significan "medido"
# segun el diccionario de RECH5; cualquier otro codigo (no presente,
# rechazo, parcial, otro) invalida el dato y se marca como NaN en vez de
# usar el valor crudo (que ademas trae codigos sentinela como 999/9999).
for col in ['HA53', 'HA40', 'HA55', 'HA13']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['hemoglobina'] = np.where(df['HA55'] == 0, df['HA53'] / 10, np.nan)
df['IMC'] = np.where(df['HA13'] == 0, df['HA40'] / 100, np.nan)
print(f"hemoglobina: {df['hemoglobina'].notna().sum():,} validas de {len(df):,}")
print(f"IMC (RECH5): {df['IMC'].notna().sum():,} validas de {len(df):,}")

# --- Atencion calificada del parto (M3A-M3D) ---
# Definicion estandar OMS de "parto con atencion calificada": medico,
# obstetra, enfermera o tecnico de enfermeria. Comadrona/partera, promotor
# de salud, familiar, nadie u otro cuentan como NO calificada.
cols_calificado = [c for c in ['M3A', 'M3B', 'M3C', 'M3D'] if c in df.columns]
for col in cols_calificado:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df['atencion_calificada_parto'] = (df[cols_calificado] == 1).any(axis=1).astype(int)
print(f"atencion_calificada_parto: {df['atencion_calificada_parto'].value_counts(dropna=False).to_dict()}")

# --- Calidad del control prenatal (M42A-E) ---
# Suma de 0 a 5: cuantos de los 5 procedimientos basicos (peso, medida de
# barriga, presion arterial, examen de orina, examen de sangre) se
# realizaron durante el control prenatal. El codigo 8 significa "no sabe"
# (no es un tercer valor valido de conteo) y se pasa a NaN antes de sumar;
# si no, se suma como si fueran 8 procedimientos y el rango sale mal (0-18
# en vez de 0-5, como paso en la primera version de este bloque).
cols_m42 = [c for c in ['M42A', 'M42B', 'M42C', 'M42D', 'M42E'] if c in df.columns]
for col in cols_m42:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].where(df[col].isin([0, 1]), np.nan)
df['control_prenatal_calidad'] = df[cols_m42].sum(axis=1, min_count=1)
print(f"control_prenatal_calidad: rango {df['control_prenatal_calidad'].min()}-{df['control_prenatal_calidad'].max()}")

# ==========================================
# PASO 6: SELECCIONAR COLUMNAS FINALES Y GUARDAR
# ==========================================

print("\n" + "=" * 70)
print("PASO 6: SELECCIONANDO COLUMNAS FINALES Y GUARDANDO")
print("=" * 70)

print(f"Dimensiones antes de seleccionar columnas: {df.shape[0]:,} filas, {df.shape[1]} columnas")

if len(df) == 0:
    print("ERROR: no quedo ningun CASEID presente en todos los modulos, no hay nada que guardar")
    exit(1)

columnas_presentes = [c for c in COLUMNAS_FINALES if c in df.columns]
columnas_faltantes = [c for c in COLUMNAS_FINALES if c not in df.columns]
if columnas_faltantes:
    print(f"ADVERTENCIA: columnas no encontradas y omitidas: {columnas_faltantes}")

df_final = df[columnas_presentes].copy()
print(f"Dimensiones finales: {df_final.shape[0]:,} filas, {df_final.shape[1]} columnas")
print(f"Columnas: {', '.join(columnas_presentes)}")

clase_0 = int((df_final['target'] == 0).sum())
clase_1 = int((df_final['target'] == 1).sum())
print("\nDistribucion de la variable objetivo:")
print(f"   CLASE 0 (Sin perdida): {clase_0:,} ({clase_0 / len(df_final) * 100:.2f}%)")
print(f"   CLASE 1 (Perdida): {clase_1:,} ({clase_1 / len(df_final) * 100:.2f}%)")

print(f"\nGuardando en: {OUTPUT_FILE}")
df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print("Archivo guardado exitosamente")

print("\n" + "=" * 70)
print("PROCESO COMPLETADO")
print("=" * 70)
print(f"Anio procesado: {ANIO}")
print(f"Registros finales: {len(df_final):,}")
print(f"Columnas finales: {df_final.shape[1]}")
print(f"Archivo de salida: {OUTPUT_FILE}")