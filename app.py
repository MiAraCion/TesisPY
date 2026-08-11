import pandas as pd
import numpy as np

inputFile = "./2025/1632/RE223132_2024.csv"
outputFile = "./2025/1632/endes_filtrada_ml.csv"

print("cargando archivo CSV...")

try:
    df = pd.read_csv(inputFile, encoding='latin-1', low_memory=False)
    print("archivo cargado correctamente...")
    print(f"DIMENSIONES ORIGINALES")
    print(f"FILAS: {df.shape[0]}")
    print(f"COLUMNAS: {df.shape[1]}")

except FileNotFoundError:
    print(f"❌ Error: No se encontro el archivo {inputFile}")
    print("⚠️  Asegurate de que el archivo este en la ruta correcta")
    exit(1)

except Exception as e:
    print(f"❌ Error al cargar el archivo: {e}")
    exit(1)

print("\n🔍 Explorando la estructura del archivo...")

# Ver los nombres de las primeras columnas
print("\n📋 Todas las columnas:")
print(df.columns.tolist())

# Ver las primeras filas para entender los datos
print("\n📊 Primeras 3 filas:")
print(df.head(3))

endes_copy = df.copy()

registros_iniciales = len(endes_copy)
print(f"\nRegistros iniciales de la copia: {registros_iniciales}")

print("==========================================================================")
print("APLICANDO FILTROS Y LIMPIEZA CON VARIABLES DEL DICCIONARIO ENDES")
print("==========================================================================")

# Convertir variables del diccionario a numerico para evitar inconsistencias de tipo
cols_clave = ['V212', 'V201', 'V213', 'V228', 'V233']
for col in cols_clave:
    if col in endes_copy.columns:
        endes_copy[col] = pd.to_numeric(endes_copy[col], errors='coerce')

# --- FILTRO 1: EDAD DE 15 A 45 AÑOS (V212) ---
f1_condicion = (endes_copy['V212'] >= 15) & (endes_copy['V212'] <= 45)
excluidos_f1 = len(endes_copy[~f1_condicion])
df_filtered = endes_copy[f1_condicion].copy()

# --- FILTRO 2: AL MENOS 1 GESTACION (V201 > 0 o V228 == 1) ---
f2_condicion = (df_filtered['V201'] > 0) | (df_filtered['V228'] == 1)
excluidos_f2 = len(df_filtered[~f2_condicion])
df_filtered = df_filtered[f2_condicion].copy()

# --- FILTRO 3: EXCLUIR EMBARAZADAS ACTUALES (V213 == 1) ---
f3_condicion = df_filtered['V213'] != 1
excluidos_f3 = len(df_filtered[~f3_condicion])
df_filtered = df_filtered[f3_condicion].copy()

# --- FILTRO 4: ELIMINAR PERDIDAS DECLARADAS CON DURACION INCONSISTENTE O DESCONOCIDA ---
# (Si V228 == 1, V233 no debe ser nulo ni tomar valores 97 o 98)
f4_inconsistentes = (df_filtered['V228'] == 1) & (
    df_filtered['V233'].isna() | df_filtered['V233'].isin([97, 98])
)
excluidos_f4 = len(df_filtered[f4_inconsistentes])
df_filtered = df_filtered[~f4_inconsistentes].copy()

# --- FILTRO 5 Y ASIGNACION DE TARGET (< 22 SEMANAS / 1-5 MESES VS EMBARAZO EXITOSO) ---
# Clase 1: Pérdida involuntaria entre 1 y 5 meses (V228 == 1 & V233 in [1, 2, 3, 4, 5])
# Clase 0: Embarazo exitoso (V228 == 0)
condiciones = [
    (df_filtered['V228'] == 1) & (df_filtered['V233'].isin([1, 2, 3, 4, 5])),
    (df_filtered['V228'] == 0)
]

df_filtered['target'] = np.select(condiciones, [1, 0], default=np.nan)

# Excluir registros con pérdidas >= 6 meses (óbitos tardíos) que quedaron como NaN en target
excluidos_f5 = len(df_filtered[df_filtered['target'].isna()])
df_final = df_filtered.dropna(subset=['target']).copy()
df_final['target'] = df_final['target'].astype(int)

# Total de excluidos
total_excluidos = registros_iniciales - len(df_final)

# Guardar el dataset limpio
df_final.to_csv(outputFile, index=False, encoding='latin-1')
print(f"\n✅ Archivo filtrado guardado en: {outputFile}")

print("\n==========================================================================")
print("RESUMEN DE FILTROS APLICADOS Y REGISTROS EXCLUIDOS")
print("==========================================================================")
print(f"1. Edad (V212) fuera del rango 15-45 años:                {excluidos_f1:,} excluidos")
print(f"2. Sin antecedentes gestacionales (V201 = 0 y V228 != 1):   {excluidos_f2:,} excluidos")
print(f"3. Mujeres actualmente embarazadas (V213 = 1):              {excluidos_f3:,} excluidos")
print(f"4. Pérdidas con meses desconocidos/inconsistentes (V233):  {excluidos_f4:,} excluidos")
print(f"5. Óbitos/Pérdidas tardías a partir del mes 6 (V233 >= 6): {excluidos_f5:,} excluidos")
print("--------------------------------------------------------------------------")
print(f"📊 TOTAL DE REGISTROS INICIALES: {registros_iniciales:,}")
print(f"🚫 TOTAL DE REGISTROS EXCLUIDOS: {total_excluidos:,}")
print(f"✨ TOTAL DE REGISTROS FINALES:   {len(df_final):,}")

print("\n==========================================================================")
print("TEST Y VERIFICACION FINAL DE CLASES PARA MACHINE LEARNING")
print("==========================================================================")

clase_0 = len(df_final[df_final['target'] == 0])
clase_1 = len(df_final[df_final['target'] == 1])

pct_clase_0 = (clase_0 / len(df_final)) * 100
pct_clase_1 = (clase_1 / len(df_final)) * 100

print(f"🟢 CLASE 0 (Embarazos exitosos / Sin pérdidas):                {clase_0:,} registros ({pct_clase_0:.2f}%)")
print(f"🔴 CLASE 1 (Pérdida gestacional involuntaria < 22 semanas):   {clase_1:,} registros ({pct_clase_1:.2f}%)")
print(f"TOTAL DATASET ML:                                             {len(df_final):,} registros")

print("\n🔍 Desglose por meses en las pérdidas gestacionales (Clase 1 - V233):")
print(df_final[df_final['target'] == 1]['V233'].value_counts().sort_index())

print("\n✅ Dataset procesado correctamente y listo para la selección de características y balanceo de clases.")