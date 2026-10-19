# -*- coding: utf-8 -*-
"""Reconstruye el indice de tablas y figuras usando la numeracion secuencial
actual del documento (Tabla 1-17, Figura 1-16), resaltado en amarillo."""

import copy
from pathlib import Path

from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.shared import Pt
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

DOC_PATH = Path(r"c:\Users\Remoto\Desktop\TesisPY\NuevosDocumentosJange\Plantilla de Tesis - ARAGON - JANGE - Final.docx")

doc = Document(DOC_PATH)

FIGURAS = [
    "Figura 1: Elaboración de Target",
    "Figura 2: Descripción de Fases",
    "Figura 3: Partición de datos de Muestra",
    "Figura 4: Importancia de las 19 variables candidatas según Boruta, ordenadas de mayor a menor",
    "Figura 5: Comparación de métricas del modelo sin SMOTE-NC frente a con SMOTE-NC",
    "Figura 6: F1-Score y AUC-ROC promedio (con barras de error) de los 5 modelos, validación cruzada 5-fold",
    "Figura 7: Comparación de los 5 modelos ya optimizados, evaluados con su umbral óptimo respectivo",
    "Figura 8: Matriz de confusión acumulada del mejor modelo (Gradient Boosting), validación cruzada 5-fold",
    "Figura 9: Gráfico SHAP Beeswarm: distribución del impacto de cada variable sobre la predicción",
    "Figura 10: Importancia global de cada variable según el valor absoluto medio de SHAP",
    "Figura 11: Explicación LIME del caso de alto riesgo (verdadero positivo)",
    "Figura 12: Gráfico SHAP Waterfall del caso de alto riesgo (verdadero positivo)",
    "Figura 13: Explicación LIME del caso de bajo riesgo (verdadero negativo)",
    "Figura 14: Gráfico SHAP Waterfall del caso de bajo riesgo (verdadero negativo)",
    "Figura 15: Explicación LIME del caso límite (falso positivo, probabilidad ≈ 0.5)",
    "Figura 16: Gráfico SHAP Waterfall del caso límite (falso positivo, probabilidad ≈ 0.5)",
]

TABLAS = [
    "Tabla 1: Métricas de modelos de Machine Learning en Salud Reproductiva",
    "Tabla 2: Variables Determinantes Sociodemográficos",
    "Tabla 3: Variables de Determinantes de Salud",
    "Tabla 4: Comparación de tipos de modelos de Machine Learning",
    "Tabla 5: Niveles de Análisis de Explicabilidad Global",
    "Tabla 6: Niveles de Análisis de Explicabilidad Local",
    "Tabla 7: Integración de Bases de Datos de la ENDES",
    "Tabla 8: Matriz de Confusión",
    "Tabla 9: Consolidado de Registros del Dataset",
    "Tabla 10: Comparación de Escalado en Modelos",
    "Tabla 11: Las 10 variables retenidas para el modelado, de un total de 19 evaluadas por Boruta",
    "Tabla 12: Distribución de clases en train antes/después de SMOTE-NC, y en test",
    "Tabla 13: Comparación de un mismo modelo (Random Forest) entrenado sin y con SMOTE-NC",
    "Tabla 14: Resultados promedio (± desviación estándar) de la validación cruzada de 5 particiones",
    "Tabla 15: Hiperparámetros y umbral de decisión óptimos encontrados para cada modelo",
    "Tabla 16: Matriz de confusión acumulada (5-fold) del modelo Gradient Boosting",
    "Tabla 17: Importancia global de las 10 variables del modelo, según el valor absoluto medio de SHAP",
]


def find_heading(texto_buscado):
    for i, para in enumerate(doc.paragraphs):
        if para.text.strip() == texto_buscado:
            return i
    return None


def limpiar_lista_vieja(inicio_heading):
    """Elimina los parrafos inmediatamente siguientes al heading que
    empiecen con 'Tabla' o 'Figura' (la lista vieja), sin tocar parrafos
    en blanco intermedios ni el siguiente heading real."""
    parrafos = doc.paragraphs
    i = inicio_heading + 1
    a_eliminar = []
    vacios_pendientes = []
    while i < len(parrafos):
        t = parrafos[i].text.strip()
        if t.startswith("Tabla") or t.startswith("Figura"):
            a_eliminar.extend(vacios_pendientes)
            a_eliminar.append(i)
            vacios_pendientes = []
        elif t == "":
            vacios_pendientes.append(i)
        else:
            break
        i += 1
    for idx in sorted(a_eliminar, reverse=True):
        p = parrafos[idx]._p
        p.getparent().remove(p)


def insertar_lista(despues_de_idx, items):
    ref_para = doc.paragraphs[despues_de_idx]
    ref = ref_para._p
    for item in items:
        new_p = copy.deepcopy(ref)
        for r in list(new_p.findall(qn('w:r'))):
            new_p.remove(r)
        ref.addnext(new_p)
        ref = new_p
        p_obj = Paragraph(new_p, ref_para._parent)
        run = p_obj.add_run(item)
        run.font.highlight_color = WD_COLOR_INDEX.YELLOW
        run.font.size = Pt(10.5)
        run.italic = False


idx_figuras = find_heading("Índice de figuras")
idx_tablas = find_heading("Índice de Tablas")
print("Indice de figuras en parrafo:", idx_figuras)
print("Indice de Tablas en parrafo:", idx_tablas)

# procesar primero el que esta mas abajo para no invalidar el indice del otro
if idx_tablas > idx_figuras:
    limpiar_lista_vieja(idx_tablas)
    insertar_lista(idx_tablas, TABLAS)
    limpiar_lista_vieja(idx_figuras)
    insertar_lista(idx_figuras, FIGURAS)
else:
    limpiar_lista_vieja(idx_figuras)
    insertar_lista(idx_figuras, FIGURAS)
    limpiar_lista_vieja(idx_tablas)
    insertar_lista(idx_tablas, TABLAS)

doc.save(DOC_PATH)
print("Indice reconstruido y guardado.")
