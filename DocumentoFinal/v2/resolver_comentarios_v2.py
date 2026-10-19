# -*- coding: utf-8 -*-
"""Resuelve la segunda tanda de comentarios del profesor (25-27 ago) en
Tesis - ARAGON - JANGE - Final.docx (carpeta v2). Texto que se ELIMINA se
marca en rojo + tachado; texto NUEVO o modificado se resalta en amarillo."""

import copy
from pathlib import Path

from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.text.run import Run
from docx.shared import Pt, RGBColor

DOC_PATH = Path(r"c:\Users\Remoto\Desktop\TesisPY\DocumentoFinal\v2\Tesis - ARAGON - JANGE - Final.docx")

doc = Document(DOC_PATH)
print(f"Documento: {len(doc.paragraphs)} parrafos, {len(doc.tables)} tablas")


def _clear_runs(paragraph_or_cell_para):
    for r in list(paragraph_or_cell_para.runs):
        r._element.getparent().remove(r._element)


def _add_run(paragraph, ref_elem, text, modo):
    """modo: 'red' (eliminado, tachado y rojo), 'yellow' (nuevo/cambiado), None (sin marca)"""
    new_elem = copy.deepcopy(ref_elem)
    paragraph._p.append(new_elem)
    run = Run(new_elem, paragraph)
    run.text = text
    if modo == 'red':
        run.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
        run.font.strike = True
        run.font.highlight_color = None
    elif modo == 'yellow':
        run.font.highlight_color = WD_COLOR_INDEX.YELLOW
        run.font.strike = False
    return run


def replace_full_paragraph(paragraph, texto_nuevo, marcar_viejo_en_rojo=True):
    """Reemplaza TODO el contenido de un parrafo: el texto viejo completo
    queda tachado en rojo (si marcar_viejo_en_rojo=True) seguido del texto
    nuevo resaltado en amarillo."""
    texto_viejo = "".join(r.text for r in paragraph.runs)
    if not paragraph.runs:
        return False
    ref_elem = paragraph.runs[0]._element
    _clear_runs(paragraph)
    if marcar_viejo_en_rojo and texto_viejo.strip():
        _add_run(paragraph, ref_elem, texto_viejo + " ", 'red')
    _add_run(paragraph, ref_elem, texto_nuevo, 'yellow')
    return True


def replace_full_cell(cell, texto_nuevo, marcar_viejo_en_rojo=True):
    p = cell.paragraphs[0]
    texto_viejo = "".join(r.text for r in p.runs) if p.runs else cell.text
    if p.runs:
        ref_elem = p.runs[0]._element
        _clear_runs(p)
    else:
        cell.text = ""
        p = cell.paragraphs[0]
        run0 = p.add_run("")
        ref_elem = run0._element
        _clear_runs(p)
    if marcar_viejo_en_rojo and texto_viejo.strip():
        _add_run(p, ref_elem, texto_viejo + " ", 'red')
    _add_run(p, ref_elem, texto_nuevo, 'yellow')
    # limpiar otros parrafos vacios remanentes en la celda (si el texto viejo tenia varios parrafos)
    for extra_p in cell.paragraphs[1:]:
        extra_p._p.getparent().remove(extra_p._p)


def highlight_replace_in_paragraph(paragraph, old_fragment, new_fragment):
    full_text = "".join(r.text for r in paragraph.runs)
    idx = full_text.find(old_fragment)
    if idx == -1:
        return False
    before, after = full_text[:idx], full_text[idx + len(old_fragment):]
    ref_elem = paragraph.runs[0]._element
    _clear_runs(paragraph)
    if before:
        _add_run(paragraph, ref_elem, before, None)
    _add_run(paragraph, ref_elem, new_fragment, 'yellow')
    if after:
        _add_run(paragraph, ref_elem, after, None)
    return True


resultados = []

# =================================================================
# 1) HE4 en la seccion 1.3.2 (parrafo 144) - reemplazo completo
# =================================================================
NUEVA_HE4 = ("HE4: La dimensión sociodemográfica concentra una mayor proporción "
             "de la importancia global según TreeSHAP que la dimensión de salud.")
ok = replace_full_paragraph(doc.paragraphs[144], NUEVA_HE4)
resultados.append(("HE4 narrativa (1.3.2)", ok))

# =================================================================
# 2) Matriz de Consistencia (Tabla 17 / Anexo 1): HE1, HE3, HE4, HE5
# =================================================================
t = doc.tables[17]
NUEVA_HE1_MATRIZ = ("HE1: El criterio híbrido de selección de variables (subconjunto Top-10 por "
                     "importancia acumulada, construido a partir de la salida de Boruta y complementado "
                     "con respaldo teórico de la literatura de salud reproductiva) mantiene una capacidad "
                     "predictiva igual o superior a la del subconjunto de 6 variables confirmadas "
                     "únicamente por Boruta, reduciendo la dimensionalidad frente a las 19 variables "
                     "candidatas sin pérdida de capacidad predictiva.")
NUEVA_HE3_MATRIZ = ("HE3: Los algoritmos basados en ensambles (XGBoost, Random Forest, Gradient Boosting) "
                     "superan en AUROC y F1-Score a los modelos tradicionales (SVM, KNN).")
NUEVA_HE5_MATRIZ = ("HE5: Las explicaciones locales obtenidas mediante LIME y SHAP Waterfall coinciden en "
                     "la dirección del efecto de las variables más influyentes (V025, V201 y V501), "
                     "evidenciando consistencia explicativa entre ambas técnicas.")

replace_full_cell(t.rows[4].cells[5], NUEVA_HE1_MATRIZ)
resultados.append(("HE1 en Matriz de Consistencia", True))
replace_full_cell(t.rows[6].cells[5], NUEVA_HE3_MATRIZ)
resultados.append(("HE3 en Matriz de Consistencia", True))
replace_full_cell(t.rows[7].cells[5], NUEVA_HE4)
resultados.append(("HE4 en Matriz de Consistencia", True))
replace_full_cell(t.rows[8].cells[5], NUEVA_HE5_MATRIZ)
resultados.append(("HE5 en Matriz de Consistencia", True))

# =================================================================
# 3) 2.5 XAI para la Interpretacion - lenguaje clinico/politica publica
# =================================================================
NUEVO_2_5 = (
    "De este modo, la XAI constituye un mecanismo relevante para traducir las estimaciones "
    "computacionales en evidencia interpretable desde un enfoque epidemiológico [5, 14]. En "
    "congruencia con estos hallazgos, se asume una postura crítica frente al desarrollo tradicional "
    "de algoritmos de Machine Learning, sobre todo porque, para que el modelo predictivo propuesto "
    "tenga utilidad como insumo para la interpretación del comportamiento del riesgo gestacional, no "
    "basta con identificar qué mujeres de la ENDES presentan mayor riesgo, sino ofrecer elementos que "
    "ayuden a comprender por qué el modelo llega a esa estimación, sin que ello implique que el modelo "
    "esté validado como herramienta de uso clínico directo o de política pública. Por consiguiente, se "
    "considera pertinente implementar Inteligencia Artificial Explicable como un componente integral, y "
    "no solo complementario, del diseño metodológico propuesto. Esta postura busca que las interacciones "
    "matemáticas de los determinantes sociodemográficos peruanos se traduzcan en un perfil de riesgo "
    "interpretable, que pueda servir como insumo exploratorio para investigaciones futuras en salud "
    "pública, sin constituir por sí solo una validación clínica ni una política de prevención "
    "personalizada."
)
for i, para in enumerate(doc.paragraphs):
    if 'apoyo a la toma de decisiones clínicas' in para.text:
        # solo reemplazar desde "De este modo, la XAI constituye..." en adelante
        full = "".join(r.text for r in para.runs)
        marcador = "De este modo, la XAI constituye"
        idx = full.find(marcador)
        if idx != -1:
            antes = full[:idx]
            ref_elem = para.runs[0]._element
            texto_viejo = full[idx:]
            _clear_runs(para)
            if antes:
                _add_run(para, ref_elem, antes, None)
            _add_run(para, ref_elem, texto_viejo + " ", 'red')
            _add_run(para, ref_elem, NUEVO_2_5, 'yellow')
            resultados.append(("2.5 lenguaje clinico/politica publica", True))
        break
else:
    resultados.append(("2.5 lenguaje clinico/politica publica", False))

# =================================================================
# 4) 4.4 Nivel - terminologia uniforme (parrafo 286)
# =================================================================
ok = highlight_replace_in_paragraph(
    doc.paragraphs[286],
    "La investigación consta de niveles predictivo y explicativo, con el fin de anticipar el riesgo reproductivo, bajo una explicación secuencial y estructurada. El objetivo consta en abordar tanto la necesidad clínica de anticipar el riesgo reproductivo",
    "La investigación se ubica en un nivel predictivo, con un componente de explicabilidad del modelo, con el fin de anticipar el riesgo reproductivo bajo un enfoque secuencial y estructurado. El objetivo aborda tanto la necesidad clínica de anticipar el riesgo reproductivo"
)
resultados.append(("4.4 Nivel terminologia uniforme", ok))

# =================================================================
# 5) 4.5 Poblacion - definir unidad de analisis (parrafo 293)
# =================================================================
NUEVA_FRASE_UNIDAD = (
    " La unidad de análisis de esta investigación es la mujer entrevistada, no el embarazo ni el "
    "episodio gestacional de forma individual: cada registro del dataset corresponde a una mujer, y "
    "la variable objetivo refleja si esa mujer reporta, en su historial reproductivo, al menos un "
    "episodio de pérdida gestacional temprana (V233 < 6), y no un evento gestacional específico "
    "monitoreado de forma prospectiva."
)
full293 = "".join(r.text for r in doc.paragraphs[293].runs)
if full293.strip():
    ref_elem = doc.paragraphs[293].runs[0]._element
    _add_run(doc.paragraphs[293], ref_elem, NUEVA_FRASE_UNIDAD, 'yellow')
    resultados.append(("4.5 Poblacion: unidad de analisis (agregado)", True))
else:
    resultados.append(("4.5 Poblacion: unidad de analisis (agregado)", False))

doc.save(DOC_PATH)

print("\n--- RESULTADOS ---")
for etiqueta, ok in resultados:
    print(f"   [{'OK' if ok else 'NO ENCONTRADO'}] {etiqueta}")
