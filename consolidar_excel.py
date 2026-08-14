import pandas as pd
from pathlib import Path

# ==========================================
# CONFIGURACION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "endes_2022_2025.xlsx"

ARCHIVOS_POR_ANIO = {
    2022: BASE_DIR / "2022" / "endes_filtrada_ml_2022.csv",
    2023: BASE_DIR / "2023" / "endes_filtrada_ml_2023.csv",
    2024: BASE_DIR / "2024" / "endes_filtrada_ml_2024.csv",
    2025: BASE_DIR / "2025" / "endes_filtrada_ml_2025.csv",
}

# ==========================================
# CARGAR CADA AÑO Y ARMAR EL CONSOLIDADO
# ==========================================

print("=" * 70)
print("CARGANDO LOS CSV DE CADA AÑO")
print("=" * 70)

dataframes_por_anio = {}
for anio, ruta in ARCHIVOS_POR_ANIO.items():
    if not ruta.exists():
        print(f"   {anio}: ARCHIVO NO ENCONTRADO ({ruta}), se omite")
        continue
    df = pd.read_csv(ruta, encoding='utf-8-sig')
    df.insert(1, 'anio', anio)
    dataframes_por_anio[anio] = df
    print(f"   {anio}: {len(df):,} filas, {df.shape[1]} columnas")

df_consolidado = pd.concat(dataframes_por_anio.values(), ignore_index=True)

print(f"\nConsolidado total: {len(df_consolidado):,} filas")
clase_0 = int((df_consolidado['target'] == 0).sum())
clase_1 = int((df_consolidado['target'] == 1).sum())
print(f"   CLASE 0 (Sin perdida): {clase_0:,} ({clase_0 / len(df_consolidado) * 100:.2f}%)")
print(f"   CLASE 1 (Perdida): {clase_1:,} ({clase_1 / len(df_consolidado) * 100:.2f}%)")

# ==========================================
# GUARDAR EN EXCEL (una hoja por año + una hoja consolidada)
# ==========================================

print("\n" + "=" * 70)
print(f"GUARDANDO EXCEL EN: {OUTPUT_FILE}")
print("=" * 70)

with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
    df_consolidado.to_excel(writer, sheet_name='Consolidado_2022_2025', index=False)
    for anio, df in dataframes_por_anio.items():
        df.to_excel(writer, sheet_name=str(anio), index=False)

print("Archivo Excel guardado exitosamente")
print(f"\nHojas creadas: Consolidado_2022_2025, {', '.join(str(a) for a in dataframes_por_anio.keys())}")
print(f"Archivo: {OUTPUT_FILE}")