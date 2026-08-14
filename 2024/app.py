import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# CONFIGURACION (especifica del anio 2024)
# ==========================================

ANIO = 2024
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "endes_filtrada_ml_2024.csv"

# Nombres de archivo tal como existen en las carpetas de este anio
ARCHIVO_1631 = "REC0111_2024.csv"
ARCHIVO_1632 = "RE223132_2024.csv"
ARCHIVO_1633 = "REC41_2024.csv"
ARCHIVO_1635 = "RE516171_2024.csv"
ARCHIVO_1640 = "CSALUD01_2024.csv"

# Columnas finales del dataset, segun la tabla de la matriz de consistencia
COLUMNAS_FINALES = [
    'CASEID',
    # --- Factores Sociodemograficos (1631) ---
    'V012', 'V106', 'V025', 'V190', 'V040',
    # --- Factores Sociodemograficos (1635) ---
    'V501',
    # --- Habitos de Salud: uso de anticonceptivos (1632) ---
    'V302', 'V313',
    # --- Antecedentes obstetricos (1632, mi preferencia) ---
    'V234', 'V222', 'V201',
    # --- Habitos de Salud: tabaco, alcohol, IMC (1640) ---
    'QS200', 'QS202', 'QS205C',
    'QS206', 'QS208', 'QS210',
    'QS900', 'QS901',
    # --- Habitos de Salud: numero de controles prenatales (1633) ---
    'M14',
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

# HHID + V003 se guardan solo para poder enlazar mas adelante con CSALUD01 (1640),
# que no tiene CASEID sino HHID + numero de linea de la persona dentro del hogar.
df_1631 = df_1631[['CASEID', 'HHID', 'V003', 'V012', 'V106', 'V025', 'V190', 'V040']]

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

# --- 1633: Numero de controles prenatales (M14) ---
# REC41 tiene una fila POR EMBARAZO (indice MIDX), no por mujer.
# Nos quedamos con el embarazo mas reciente (MIDX == 1) para tener 1 fila por CASEID.
print("\nModulo 1633 (REC41) - Numero de controles prenatales")
ruta_rec41 = BASE_DIR / "1633" / ARCHIVO_1633
df_rec41 = cargar_csv(ruta_rec41)
verificar_anio(df_rec41, ARCHIVO_1633)
df_rec41 = df_rec41[df_rec41['MIDX'] == 1][['CASEID', 'M14']]
df_rec41 = df_rec41[df_rec41['CASEID'].isin(caseids_filtrados)]

antes = df['CASEID'].nunique()
df = df.merge(df_rec41, on='CASEID', how='inner')
print(f"   CASEID: {antes:,} -> {df['CASEID'].nunique():,}")

# --- 1640: Habitos de salud (tabaco, alcohol, IMC) ---
# CSALUD01 NO tiene CASEID (es un cuestionario de hogar, no individual).
# Se enlaza por HHID + numero de linea de la persona dentro del hogar (QSNUMERO == V003).
print("\nModulo 1640 (CSALUD01) - Habitos de salud")
print("   (Enlazado por HHID + linea de persona, no por CASEID)")
ruta_csalud01 = BASE_DIR / "1640_cuestionario_del_hogar" / ARCHIVO_1640
df_csalud = cargar_csv(ruta_csalud01)
verificar_anio(df_csalud, ARCHIVO_1640)

cols_habitos = ['QS200', 'QS202', 'QS205C', 'QS206', 'QS208', 'QS210', 'QS900', 'QS901']
cols_habitos = [c for c in cols_habitos if c in df_csalud.columns]
df_csalud = df_csalud[['HHID', 'QSNUMERO'] + cols_habitos]
df_csalud = df_csalud.rename(columns={'QSNUMERO': 'V003'})

antes = df['CASEID'].nunique()
df = df.merge(df_csalud, on=['HHID', 'V003'], how='inner')
print(f"   CASEID: {antes:,} -> {df['CASEID'].nunique():,}")
print("   NOTA: CSALUD01 solo se aplica a una submuestra de hogares en ENDES,")
print("   por eso esta union reduce el N de forma esperada (no es un error).")

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