# -*- coding: utf-8 -*-
"""Llena 5. Resultados (contenido completo, organizado por objetivo especifico),
y deja una guia estructurada (esqueleto resaltado) en 6. Discusion y 7. Conclusiones."""

import copy
from pathlib import Path

from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.shared import Pt

DOC_PATH = Path(r"c:\Users\Remoto\Desktop\TesisPY\DocumentoFinal\v2\Tesis - ARAGON - JANGE - Final.docx")

doc = Document(DOC_PATH)
print(f"Documento: {len(doc.paragraphs)} parrafos, {len(doc.tables)} tablas")


def find_heading(texto):
    for i, para in enumerate(doc.paragraphs):
        if para.text.strip() == texto:
            return para
    raise ValueError(f"No se encontro heading: {texto!r}")


def insert_after(ref_para, style_name, text, italic=False):
    """Inserta un nuevo parrafo justo despues de ref_para, con el estilo dado,
    resaltado en amarillo. Devuelve el nuevo parrafo (para encadenar)."""
    new_p = copy.deepcopy(ref_para._p)
    for r in list(new_p.findall(qn('w:r'))):
        new_p.remove(r)
    ref_para._p.addnext(new_p)
    p_obj = Paragraph(new_p, ref_para._parent)
    p_obj.style = doc.styles[style_name]
    run = p_obj.add_run(text)
    run.font.highlight_color = WD_COLOR_INDEX.YELLOW
    run.italic = italic
    return p_obj


# =================================================================
# 5. RESULTADOS
# =================================================================
cursor = find_heading("5. Resultados")

cursor = insert_after(cursor, "Normal",
    "Los resultados se presentan organizados según los cinco objetivos específicos de la "
    "investigación, remitiendo a las tablas y figuras detalladas en la sección 4.10 (Desarrollo "
    "de la solución) para el procedimiento técnico completo.")

cursor = insert_after(cursor, "Heading 2", "5.1 Resultados de la selección de características (OE1)")
cursor = insert_after(cursor, "Normal",
    "De las 19 variables candidatas evaluadas, el algoritmo Boruta confirmó estadísticamente 6 "
    "como relevantes: paridad (V201), intervalo intergenésico (V222), altitud del conglomerado "
    "(V040), edad (V012), índice de masa corporal (IMC) e índice de riqueza (V190) (ver Tabla 11 y "
    "Figura 4, sección 4.10.3). Dado que estas 6 variables por sí solas dejarían fuera predictores "
    "con respaldo teórico documentado en el marco teórico, se adoptó un criterio híbrido: las 10 "
    "variables de mayor importancia acumulada (88.3% del total), que además de las 6 confirmadas "
    "incorporan área de residencia (V025), nivel educativo (V106), estado civil (V501) y consumo "
    "de alcohol (QS206). La comparación empírica entre ambos conjuntos mostró un desempeño "
    "prácticamente equivalente entre las 6 variables confirmadas y las 10 del criterio híbrido, con "
    "solo Gradient Boosting mostrando una mejora marginal (F1 +0.0002) al usar el conjunto ampliado; "
    "en Random Forest, SVM y KNN el AUROC fue incluso mayor usando únicamente las 6 confirmadas.")

cursor = insert_after(cursor, "Heading 2", "5.2 Resultados del balanceo de clases (OE2)")
cursor = insert_after(cursor, "Normal",
    "La aplicación de SMOTE-NC sobre el conjunto de entrenamiento incrementó la sensibilidad "
    "(Recall) y el F1-Score en los cinco algoritmos evaluados, sin excepción (ver Tabla 12-13 y "
    "Figura 5, sección 4.10.4). Sin embargo, el efecto sobre la capacidad de discriminación no fue "
    "uniforme: el PR-AUC disminuyó de forma consistente en los 5 modelos, y el AUROC disminuyó en "
    "los modelos de ensamble basados en árboles (Random Forest, Gradient Boosting, XGBoost) pero "
    "aumentó en los modelos no basados en árboles (KNN, SVM). Este resultado sostiene HE2 en lo "
    "relativo a Recall y F1, y evidencia que el costo del balanceo sobre la capacidad de "
    "discriminación depende de la familia de algoritmo.")

cursor = insert_after(cursor, "Heading 2", "5.3 Resultados de la comparación de modelos (OE3)")
cursor = insert_after(cursor, "Normal",
    "Bajo validación cruzada estratificada de 5 particiones, Gradient Boosting resultó el modelo "
    "óptimo, con F1=0.2114±0.006, AUROC=0.6499±0.0111 y PR-AUC=0.2050±0.0128, seguido de XGBoost, "
    "Random Forest, SVM y KNN (ver Tabla 14, sección 4.10.5-4.10.6). La prueba de significancia "
    "estadística mediante bootstrap no paramétrico (B=2,000) confirmó que los 5 modelos discriminan "
    "mejor que el azar: el límite inferior del intervalo de confianza al 95% del AUROC superó 0.5 en "
    "todos los casos (desde 0.566 en KNN hasta 0.634 en Gradient Boosting). No obstante, las "
    "diferencias de rendimiento entre modelos son pequeñas, por lo que no puede sostenerse una "
    "superioridad estadísticamente contundente de los ensambles sobre los modelos tradicionales, "
    "solo una tendencia numérica favorable.")

cursor = insert_after(cursor, "Heading 2", "5.4 Resultados de explicabilidad global (OE4)")
cursor = insert_after(cursor, "Normal",
    "El análisis TreeSHAP identificó al área de residencia (V025) como la variable de mayor peso "
    "predictivo, concentrando el 24.6% de la importancia total, seguida del estado civil (V501, "
    "20.5%) y el nivel educativo (V106, 13.8%) (ver Tabla 17 y Figuras 9-10, sección 4.10.7). "
    "Agrupando por dimensión, los determinantes sociodemográficos concentran el 71.5% de la "
    "importancia total del modelo, frente a un 28.5% de los determinantes de salud.")

cursor = insert_after(cursor, "Heading 2", "5.5 Resultados de explicabilidad local (OE5)")
cursor = insert_after(cursor, "Normal",
    "En los tres casos analizados mediante LIME y SHAP Waterfall (alto riesgo, bajo riesgo y un "
    "caso límite), ambas técnicas coincidieron en la dirección del efecto de las variables más "
    "influyentes: área de residencia (V025), paridad (V201) y estado civil (V501) (ver Figuras "
    "11-16, sección 4.10.7). El caso límite —un falso positivo con probabilidad predicha de "
    "0.5000— resultó especialmente informativo al mostrar un perfil sin variables individuales con "
    "contribución dominante, ilustrando el tipo de caso donde el modelo tiene mayor probabilidad de "
    "error.")

cursor = insert_after(cursor, "Heading 2", "5.6 Síntesis de contraste de hipótesis")
cursor = insert_after(cursor, "Normal",
    "La Tabla 18 resume el grado de sustento empírico obtenido para cada hipótesis específica.")

# tabla de sintesis de hipotesis (tabla nueva, independiente, luego se reubica)
nueva_tabla = doc.add_table(rows=6, cols=3)
nueva_tabla.style = 'Table Grid'
tbl_elem = nueva_tabla._tbl
tbl_elem.getparent().remove(tbl_elem)
cursor._p.addnext(tbl_elem)

datos_hipotesis = [
    ("Hipótesis", "Resultado", "Evidencia"),
    ("HE1 (criterio híbrido Top-10)", "Parcialmente sostenida",
     "Desempeño comparable entre 6 confirmadas y Top-10; sin superioridad clara del Top-10"),
    ("HE2 (SMOTE-NC)", "Sostenida (Recall/F1); efecto mixto en AUROC/PR-AUC",
     "Recall y F1 suben en los 5 modelos; PR-AUC baja en los 5; AUROC mixto por familia de algoritmo"),
    ("HE3 (ensambles vs. tradicionales)", "Sostenida, con matiz",
     "Ensambles superan numéricamente en AUROC/F1, pero con diferencias pequeñas"),
    ("HE4 (dimensión sociodemográfica vs. salud)", "Sostenida",
     "71.5% vs. 28.5% de la importancia global según TreeSHAP"),
    ("HE5 (consistencia LIME/SHAP)", "Sostenida",
     "Coincidencia de dirección en V025, V201 y V501 en los 3 casos analizados"),
]
for i, fila in enumerate(datos_hipotesis):
    for j, val in enumerate(fila):
        cell = nueva_tabla.rows[i].cells[j]
        cell.text = val
        for r in cell.paragraphs[0].runs:
            r.font.size = Pt(9)
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
            if i == 0:
                r.bold = True

doc.save(DOC_PATH)
print("5. Resultados completado.")
