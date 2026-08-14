# TesisPY
# TesisPY
# TesisPY


C:\Users\Remoto\Desktop\TesisPY>py app.py
======================================================================
PASO 1: CARGANDO MODULO PRINCIPAL (1632 - Historia de Nacimiento)
======================================================================
Cargando: 2025\1632\RE223132_2024.csv
Archivo cargado correctamente
Dimensiones: 34252 filas, 149 columnas

======================================================================
PASO 2: APLICANDO FILTROS EN MODULO 1632
======================================================================

Registros iniciales: 34,252

FILTRO 1: Edad de 15 a 49 años (V212)
   Excluidos: 10,150 | Quedan: 24,102

FILTRO 2: Al menos 1 gestacion (V201 > 0 o V228 == 1)
   Excluidos: 0 | Quedan: 24,102

FILTRO 3: Excluir embarazadas actuales (V213 == 1)
   Excluidos: 537 | Quedan: 23,565

FILTRO 4: Eliminar perdidas con duracion inconsistente (V233)
   Excluidos: 3,174 | Quedan: 20,391

FILTRO 5: Crear variable objetivo 'target'
   Excluidos (perdidas tardias): 119 | Quedan: 20,272

======================================================================
RESUMEN DE FILTROS APLICADOS
======================================================================
Total inicial: 34,252
Excluidos (edad): 10,150
Excluidos (sin gestacion): 0
Excluidos (embarazadas): 537
Excluidos (duracion inconsistente): 3,174
Excluidos (perdidas tardias): 119
Quedan: 20,272

======================================================================
PASO 3: EXTRAYENDO CASEID DE LOS CASOS FILTRADOS
======================================================================
CASEID unicos encontrados: 20,272

======================================================================
PASO 4: BUSCANDO DATOS EN OTROS MODULOS
======================================================================

Procesando modulo 1631: Datos Basicos de MEF
   Cargando: REC0111_2025.csv
      Encontrados: 385 registros
      Eliminando columnas duplicadas: ï»¿ID1...
      Unido exitosamente. Dimensiones: (20272, 251)
   Cargando: REC91_2025.csv
      Encontrados: 385 registros
      Eliminando columnas duplicadas: ï»¿ID1...
      Unido exitosamente. Dimensiones: (20272, 592)

Procesando modulo 1633: Embarazo, Parto, Puerperio
   Cargando: REC41_2025.csv
      Encontrados: 227 registros
      Eliminando columnas duplicadas: ï»¿ID1...
      Unido exitosamente. Dimensiones: (20302, 737)
   Cargando: REC94_2025.csv
      Encontrados: 227 registros
      Eliminando columnas duplicadas: ï»¿ID1...
      Unido exitosamente. Dimensiones: (20364, 796)

Procesando modulo 1635: Nupcialidad y Fecundidad
   Cargando: RE516171_2025.csv
      Encontrados: 351 registros
      Eliminando columnas duplicadas: ï»¿ID1...
      Unido exitosamente. Dimensiones: (20364, 880)

Procesando modulo 1637: Mortalidad Materna
   Cargando: REC83_2025.csv
      Encontrados: 1,672 registros
      Eliminando columnas duplicadas: ï»¿ID1...
      Unido exitosamente. Dimensiones: (22131, 897)
   Cargando: REC84DV_2025.csv
      Encontrados: 351 registros
      Eliminando columnas duplicadas: ï»¿ID1...
      Unido exitosamente. Dimensiones: (22131, 1116)

======================================================================
PASO 5: GUARDANDO ARCHIVO FINAL
======================================================================
Dimensiones finales: 22,131 filas, 1,116 columnas

Distribucion de la variable objetivo:
   CLASE 0 (Sin perdida): 20,376 (92.07%)
   CLASE 1 (Perdida): 1,755 (7.93%)

Guardando en: ./2025/endes_filtrada_ml_completa.csv
Archivo guardado exitosamente

======================================================================
REPORTE FINAL DE PROCESAMIENTO
======================================================================
Modulos procesados:
   1631: REC0111_2025.csv, REC91_2025.csv
   1633: REC41_2025.csv, REC94_2025.csv
   1635: RE516171_2025.csv
   1637: REC83_2025.csv, REC84DV_2025.csv

Registros finales: 22,131
Columnas finales: 1116

Archivo de salida: ./2025/endes_filtrada_ml_completa.csv

PROCESO COMPLETADO

C:\Users\Remoto\Desktop\TesisPY>