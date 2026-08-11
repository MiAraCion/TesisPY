import pandas as pd
import numpy as np
import os
from pathlib import Path
import glob

# ==========================================
# CONFIGURACION
# ==========================================

BASE_DIR = "./2025"
OUTPUT_FILE = "./2025/endes_filtrada_ml_completa.csv"

# Diccionario de modulos a procesar
# Clave: codigo del modulo, Valor: archivo CSV a buscar
MODULOS = {
    '1631': {
        'carpeta': '1631',
        'archivos': ['REC0111_2025.csv', 'REC91_2025.csv'],  # Datos Basicos de MEF
        'descripcion': 'Datos Basicos de MEF'
    },
    '1632': {
        'carpeta': '1632',
        'archivos': ['RE223132_2024.csv'],  # Historia de Nacimiento
        'descripcion': 'Historia de Nacimiento (PRINCIPAL)'
    },
    '1633': {
        'carpeta': '1633',
        'archivos': ['REC41_2025.csv', 'REC94_2025.csv'],  # Embarazo, Parto, Puerperio
        'descripcion': 'Embarazo, Parto, Puerperio'
    },
    '1635': {
        'carpeta': '1635',
        'archivos': ['RE516171_2025.csv'],  # Nupcialidad - Fecundidad
        'descripcion': 'Nupcialidad y Fecundidad'
    },
    '1637': {
        'carpeta': '1637',
        'archivos': ['REC83_2025.csv', 'REC84DV_2025.csv'],  # Mortalidad Materna
        'descripcion': 'Mortalidad Materna'
    }
}

# ==========================================
# FUNCION PARA CARGAR ARCHIVO CON DETECCION DE CODIFICACION
# ==========================================

def cargar_csv(ruta_archivo):
    """Carga un CSV intentando diferentes codificaciones"""
    try:
        # Intentar con latin-1 primero (comun en ENDES)
        df = pd.read_csv(ruta_archivo, encoding='latin-1', low_memory=False)
        return df
    except UnicodeDecodeError:
        try:
            # Si falla, intentar con utf-8
            df = pd.read_csv(ruta_archivo, encoding='utf-8', low_memory=False)
            return df
        except Exception as e:
            print(f"   Error al cargar {ruta_archivo}: {e}")
            return None
    except Exception as e:
        print(f"   Error al cargar {ruta_archivo}: {e}")
        return None

# ==========================================
# 1. CARGAR MODULO PRINCIPAL (1632)
# ==========================================

print("="*70)
print("PASO 1: CARGANDO MODULO PRINCIPAL (1632 - Historia de Nacimiento)")
print("="*70)

ruta_1632 = Path(BASE_DIR) / "1632" / "RE223132_2024.csv"

if not ruta_1632.exists():
    print(f"ERROR: No se encontro el archivo {ruta_1632}")
    print("Verifica que el archivo este en la ruta correcta")
    exit(1)

print(f"Cargando: {ruta_1632}")
df_1632 = cargar_csv(ruta_1632)

if df_1632 is None:
    print("ERROR: No se pudo cargar el archivo principal")
    exit(1)

print("Archivo cargado correctamente")
print(f"Dimensiones: {df_1632.shape[0]} filas, {df_1632.shape[1]} columnas")

# ==========================================
# 2. APLICAR FILTROS EN MODULO 1632
# ==========================================

print("\n" + "="*70)
print("PASO 2: APLICANDO FILTROS EN MODULO 1632")
print("="*70)

# Crear copia para no modificar el original
df_filtered = df_1632.copy()
registros_iniciales = len(df_filtered)

print(f"\nRegistros iniciales: {registros_iniciales:,}")

# --- CONVERTIR A NUMERICO ---
cols_clave = ['V212', 'V201', 'V213', 'V228', 'V233']
for col in cols_clave:
    if col in df_filtered.columns:
        df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')

# --- FILTRO 1: EDAD DE 15 A 49 AÑOS (V212) ---
print("\nFILTRO 1: Edad de 15 a 49 años (V212)")
f1_condicion = (df_filtered['V212'] >= 15) & (df_filtered['V212'] <= 49)
excluidos_f1 = len(df_filtered[~f1_condicion])
df_filtered = df_filtered[f1_condicion].copy()
print(f"   Excluidos: {excluidos_f1:,} | Quedan: {len(df_filtered):,}")

# --- FILTRO 2: AL MENOS 1 GESTACION (V201 > 0 o V228 == 1) ---
print("\nFILTRO 2: Al menos 1 gestacion (V201 > 0 o V228 == 1)")
f2_condicion = (df_filtered['V201'] > 0) | (df_filtered['V228'] == 1)
excluidos_f2 = len(df_filtered[~f2_condicion])
df_filtered = df_filtered[f2_condicion].copy()
print(f"   Excluidos: {excluidos_f2:,} | Quedan: {len(df_filtered):,}")

# --- FILTRO 3: EXCLUIR EMBARAZADAS ACTUALES (V213 == 1) ---
print("\nFILTRO 3: Excluir embarazadas actuales (V213 == 1)")
f3_condicion = df_filtered['V213'] != 1
excluidos_f3 = len(df_filtered[~f3_condicion])
df_filtered = df_filtered[f3_condicion].copy()
print(f"   Excluidos: {excluidos_f3:,} | Quedan: {len(df_filtered):,}")

# --- FILTRO 4: ELIMINAR PERDIDAS CON DURACION INCONSISTENTE ---
print("\nFILTRO 4: Eliminar perdidas con duracion inconsistente (V233)")
f4_inconsistentes = (df_filtered['V228'] == 1) & (
    df_filtered['V233'].isna() | df_filtered['V233'].isin([97, 98])
)
excluidos_f4 = len(df_filtered[f4_inconsistentes])
df_filtered = df_filtered[~f4_inconsistentes].copy()
print(f"   Excluidos: {excluidos_f4:,} | Quedan: {len(df_filtered):,}")

# --- FILTRO 5: CREAR TARGET (VARIABLE OBJETIVO) ---
print("\nFILTRO 5: Crear variable objetivo 'target'")
condiciones = [
    (df_filtered['V228'] == 1) & (df_filtered['V233'].isin([1, 2, 3, 4, 5])),  # Perdida < 6 meses
    (df_filtered['V228'] == 0)  # Sin perdida
]

df_filtered['target'] = np.select(condiciones, [1, 0], default=np.nan)

# Excluir perdidas tardias (> 6 meses)
excluidos_f5 = len(df_filtered[df_filtered['target'].isna()])
df_filtered = df_filtered.dropna(subset=['target']).copy()
df_filtered['target'] = df_filtered['target'].astype(int)
print(f"   Excluidos (perdidas tardias): {excluidos_f5:,} | Quedan: {len(df_filtered):,}")

# --- RESUMEN DE FILTROS ---
print("\n" + "="*70)
print("RESUMEN DE FILTROS APLICADOS")
print("="*70)
print(f"Total inicial: {registros_iniciales:,}")
print(f"Excluidos (edad): {excluidos_f1:,}")
print(f"Excluidos (sin gestacion): {excluidos_f2:,}")
print(f"Excluidos (embarazadas): {excluidos_f3:,}")
print(f"Excluidos (duracion inconsistente): {excluidos_f4:,}")
print(f"Excluidos (perdidas tardias): {excluidos_f5:,}")
print(f"Quedan: {len(df_filtered):,}")

# ==========================================
# 3. EXTRAER CASEID UNICOS
# ==========================================

print("\n" + "="*70)
print("PASO 3: EXTRAYENDO CASEID DE LOS CASOS FILTRADOS")
print("="*70)

if 'CASEID' not in df_filtered.columns:
    print("ERROR: No se encontro la columna CASEID en el modulo 1632")
    print(f"Columnas disponibles: {df_filtered.columns.tolist()[:10]}...")
    exit(1)

caseids_filtrados = df_filtered['CASEID'].unique().tolist()
print(f"CASEID unicos encontrados: {len(caseids_filtrados):,}")

# ==========================================
# 4. BUSCAR Y UNIR DATOS DE OTROS MODULOS
# ==========================================

print("\n" + "="*70)
print("PASO 4: BUSCANDO DATOS EN OTROS MODULOS")
print("="*70)

# Inicializar con el modulo filtrado
df_completo = df_filtered.copy()

# Diccionario para llevar registro de que se unio
modulos_cargados = {}

for codigo, info in MODULOS.items():
    if codigo == '1632':
        continue  # Ya lo procesamos
    
    print(f"\nProcesando modulo {codigo}: {info['descripcion']}")
    carpeta = Path(BASE_DIR) / info['carpeta']
    
    if not carpeta.exists():
        print(f"   Carpeta no encontrada: {carpeta}")
        continue
    
    # Buscar archivos en la carpeta
    archivos_encontrados = []
    for patron in info['archivos']:
        archivo = carpeta / patron
        if archivo.exists():
            archivos_encontrados.append(archivo)
    
    if not archivos_encontrados:
        print(f"   No se encontraron archivos para el modulo {codigo}")
        continue
    
    # Cargar y procesar cada archivo
    for archivo in archivos_encontrados:
        print(f"   Cargando: {archivo.name}")
        df_modulo = cargar_csv(archivo)
        
        if df_modulo is None:
            print(f"      Error al cargar {archivo.name}")
            continue
        
        # Verificar si tiene CASEID
        if 'CASEID' not in df_modulo.columns:
            print(f"      {archivo.name} no tiene CASEID, ignorando")
            continue
        
        # Filtrar solo los CASEID que nos interesan
        df_modulo_filtrado = df_modulo[df_modulo['CASEID'].isin(caseids_filtrados)]
        
        if len(df_modulo_filtrado) == 0:
            print(f"      No se encontraron CASEID coincidentes en {archivo.name}")
            continue
        
        print(f"      Encontrados: {len(df_modulo_filtrado):,} registros")
        
        # Eliminar columnas duplicadas (excepto CASEID y target)
        cols_a_eliminar = [col for col in df_modulo_filtrado.columns 
                          if col in df_completo.columns and col != 'CASEID' and col != 'target']
        
        if cols_a_eliminar:
            print(f"      Eliminando columnas duplicadas: {', '.join(cols_a_eliminar[:5])}...")
            df_modulo_filtrado = df_modulo_filtrado.drop(columns=cols_a_eliminar)
        
        # Unir con el dataframe principal
        df_completo = df_completo.merge(df_modulo_filtrado, on='CASEID', how='left')
        
        # Registrar que se cargó
        if codigo not in modulos_cargados:
            modulos_cargados[codigo] = []
        modulos_cargados[codigo].append(archivo.name)
        
        print(f"      Unido exitosamente. Dimensiones: {df_completo.shape}")

# ==========================================
# 5. GUARDAR ARCHIVO FINAL
# ==========================================

print("\n" + "="*70)
print("PASO 5: GUARDANDO ARCHIVO FINAL")
print("="*70)

print(f"Dimensiones finales: {df_completo.shape[0]:,} filas, {df_completo.shape[1]:,} columnas")

# Verificar distribucion de target
clase_0 = len(df_completo[df_completo['target'] == 0])
clase_1 = len(df_completo[df_completo['target'] == 1])
pct_clase_0 = (clase_0 / len(df_completo)) * 100
pct_clase_1 = (clase_1 / len(df_completo)) * 100

print("\nDistribucion de la variable objetivo:")
print(f"   CLASE 0 (Sin perdida): {clase_0:,} ({pct_clase_0:.2f}%)")
print(f"   CLASE 1 (Perdida): {clase_1:,} ({pct_clase_1:.2f}%)")

# Guardar archivo
print(f"\nGuardando en: {OUTPUT_FILE}")
try:
    df_completo.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print("Archivo guardado exitosamente")
except Exception as e:
    print(f"Error al guardar: {e}")
    exit(1)

# ==========================================
# 6. REPORTE FINAL
# ==========================================

print("\n" + "="*70)
print("REPORTE FINAL DE PROCESAMIENTO")
print("="*70)
print(f"Modulos procesados:")
for codigo, archivos in modulos_cargados.items():
    print(f"   {codigo}: {', '.join(archivos)}")

print(f"\nRegistros finales: {len(df_completo):,}")
print(f"Columnas finales: {df_completo.shape[1]}")
print(f"\nArchivo de salida: {OUTPUT_FILE}")

print("\nPROCESO COMPLETADO")