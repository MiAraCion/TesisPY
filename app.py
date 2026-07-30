import pandas as pd
import numpy as np


inputFile = "./2025/1632/RE223132_2024.csv"

outputFile = "./2025/1632/endes_filtrada_ml.csv"

print("cargando archivo CSV")

try:

    df = pd.read_csv(inputFile, encoding='latin-1')
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
print("\n📊 Primeras 3 filas:" )
print(df.head(3))


endes_copy = df.copy()

registros_iniciales = len(endes_copy)
print(f"Registros iniciales de la copia: {registros_iniciales}")


print("==========================================================================")
print("APLICANDO FILTROS")
print("==========================================================================")

