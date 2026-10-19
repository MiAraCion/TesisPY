# -*- coding: utf-8 -*-
"""Inserta la guia-esqueleto (subtitulos + bullets orientadores) en
6. Discusion y 7. Conclusiones, resaltada en amarillo, marcada como guia
a desarrollar (no es texto final)."""

import copy
from pathlib import Path

from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

DOC_PATH = Path(r"c:\Users\Remoto\Desktop\TesisPY\DocumentoFinal\v2\Tesis - ARAGON - JANGE - Final.docx")

doc = Document(DOC_PATH)
print(f"Documento: {len(doc.paragraphs)} parrafos, {len(doc.tables)} tablas")


def find_heading(texto):
    for para in doc.paragraphs:
        if para.text.strip() == texto:
            return para
    raise ValueError(f"No se encontro heading: {texto!r}")


def insert_after(ref_para, style_name, text, italic=False, bullet=False):
    new_p = copy.deepcopy(ref_para._p)
    for r in list(new_p.findall(qn('w:r'))):
        new_p.remove(r)
    # quitar numeracion/estilo de lista heredado si lo hubiera, luego se aplica el propio
    ref_para._p.addnext(new_p)
    p_obj = Paragraph(new_p, ref_para._parent)
    p_obj.style = doc.styles[style_name]
    prefix = "• " if bullet else ""
    run = p_obj.add_run(prefix + text)
    run.font.highlight_color = WD_COLOR_INDEX.YELLOW
    run.italic = italic
    return p_obj


# =================================================================
# 6. DISCUSION - guia esqueleto
# =================================================================
cursor = find_heading("6. Discusión")

cursor = insert_after(cursor, "Normal",
    "[GUÍA A DESARROLLAR — este capítulo debe contrastar cada hallazgo de la sección 5 con los "
    "estudios revisados en el Estado del Arte (capítulo 2), explicando coincidencias, diferencias y "
    "sus posibles causas. Los subtítulos y puntos orientadores de abajo son un esqueleto: "
    "reemplazar por la redacción final.]", italic=True)

guia_discusion = [
    ("6.1 Sobre la selección de características (OE1)", [
        "Contrastar con Tesfa et al. [7], que también usa Boruta sobre encuestas DHS: ¿por qué aquí se "
        "confirmaron solo 6 de 19 variables? ¿Qué dice esto sobre la señal disponible en ENDES frente "
        "a otras encuestas?",
        "Discutir la decisión de usar el criterio híbrido de 10 variables pese a que el desempeño fue "
        "comparable con solo 6 confirmadas: ¿se justifica seguir usándolo, o el hallazgo sugiere "
        "priorizar el criterio estadístico puro en investigaciones futuras?",
    ]),
    ("6.2 Sobre el balanceo de clases (OE2)", [
        "Comparar con estudios que reportan SMOTE como una mejora sin contrapartida [1, 7, 14]: el "
        "hallazgo del efecto mixto (PR-AUC y, en árboles, AUROC bajan) no suele documentarse "
        "explícitamente en la literatura revisada — presentarlo como un aporte diferenciador de esta "
        "tesis.",
    ]),
    ("6.3 Sobre el rendimiento comparativo de los modelos (OE3)", [
        "Explicar por qué el AUROC obtenido (~0.65) es más modesto que el reportado en los estudios "
        "internacionales citados (0.84–0.93): ausencia de variables clínicas/de laboratorio en ENDES, "
        "naturaleza de encuesta de hogares frente a registro clínico.",
        "Discutir si la diferencia numérica entre ensambles y modelos tradicionales (HE3) tiene "
        "relevancia práctica más allá de la estadística.",
    ]),
    ("6.4 Sobre la explicabilidad global (OE4)", [
        "Contrastar el dominio de las variables sociodemográficas (71.5%) frente a las de salud (28.5%) "
        "con lo reportado en estudios peruanos y regionales [10, 12].",
        "Discutir la divergencia entre el ranking de Boruta (impureza) y el de TreeSHAP (valores de "
        "Shapley), citando Strobl et al. (2007) sobre el sesgo de la importancia por impureza.",
    ]),
    ("6.5 Sobre la explicabilidad local (OE5)", [
        "Reflexionar sobre el valor práctico del caso límite (falso positivo, probabilidad 0.5000) como "
        "hallazgo cualitativo para el personal de salud.",
        "Reconocer la limitación de generalizar a partir de solo 3 casos analizados.",
    ]),
    ("6.6 Limitaciones metodológicas transversales", [
        "Diseño transversal con integración de 4 rondas ENDES (pooling, no seguimiento longitudinal).",
        "Redefinición del desenlace como 'pérdida gestacional' (sin 'involuntaria'), dado que ENDES no "
        "distingue espontánea de inducida [48].",
        "Desfase temporal entre el evento (pérdida) y la entrevista (mediana ~35 meses) y su efecto "
        "sobre variables de 'estado actual'.",
        "Representatividad: la muestra analítica no aplica los pesos de expansión ni el diseño muestral "
        "complejo de ENDES.",
        "Exclusión de M14 (controles prenatales) por fuga temporal, y sus implicancias para variables "
        "similares no evaluadas.",
    ]),
]

for titulo, puntos in guia_discusion:
    cursor = insert_after(cursor, "Heading 2", titulo)
    for punto in puntos:
        cursor = insert_after(cursor, "Normal", punto, bullet=True)

# =================================================================
# 7. CONCLUSIONES - guia esqueleto
# =================================================================
cursor = find_heading("7. Conclusiones")

cursor = insert_after(cursor, "Normal",
    "[GUÍA A DESARROLLAR — cada conclusión debe responder directamente al objetivo correspondiente, "
    "sustentada exclusivamente en los resultados obtenidos (sección 5), sin introducir citas ni "
    "literatura nueva. Reemplazar los puntos orientadores por la redacción final.]", italic=True)

guia_conclusiones = [
    ("7.1 Conclusión general", [
        "Responder directamente al problema y objetivo general: ¿en qué medida el modelamiento "
        "predictivo y explicable permitió estimar el riesgo de pérdida gestacional? Sintetizar el "
        "resultado del modelo óptimo (Gradient Boosting, AUROC=0.6499, significativo vs. azar) y su "
        "alcance real (discriminación moderada, no un instrumento clínico validado).",
    ]),
    ("7.2 Conclusión OE1 (Boruta)", [
        "¿Se logró optimizar el espacio de características? Mencionar las 6 confirmadas, las 19 "
        "candidatas, y el hallazgo de que el criterio híbrido no aportó una mejora consistente frente "
        "al conjunto puramente confirmado.",
    ]),
    ("7.3 Conclusión OE2 (SMOTE-NC)", [
        "¿Se mitigó el sesgo de clase? Mencionar la mejora en Recall/F1 en los 5 modelos y el matiz del "
        "efecto mixto sobre AUROC/PR-AUC.",
    ]),
    ("7.4 Conclusión OE3 (Benchmark)", [
        "¿Cuál algoritmo presentó el mejor rendimiento? Gradient Boosting, con las métricas obtenidas; "
        "precisar que la diferencia frente a los demás modelos es moderada.",
    ]),
    ("7.5 Conclusión OE4 (SHAP global)", [
        "¿Qué determinantes pesan más y en qué dirección? Área de residencia, estado civil y nivel "
        "educativo; predominio de la dimensión sociodemográfica.",
    ]),
    ("7.6 Conclusión OE5 (LIME/SHAP local)", [
        "¿Qué aportó la explicabilidad local? Consistencia entre técnicas en los casos analizados, y el "
        "valor del caso límite para entender los errores del modelo.",
    ]),
    ("7.7 Recomendaciones para investigaciones futuras (opcional, separado de las conclusiones)", [
        "Explorar módulos de calendario reproductivo que permitan vincular cada variable al momento "
        "específico del embarazo evaluado.",
        "Incorporar el diseño muestral complejo y los pesos de expansión de ENDES si se busca "
        "inferencia poblacional.",
        "Replicar el criterio de las 6 variables confirmadas por Boruta como conjunto principal, dado "
        "que igualó el desempeño del criterio híbrido con menor dimensionalidad.",
    ]),
]

for titulo, puntos in guia_conclusiones:
    cursor = insert_after(cursor, "Heading 2", titulo)
    for punto in puntos:
        cursor = insert_after(cursor, "Normal", punto, bullet=True)

doc.save(DOC_PATH)
print("Guia de Discusion y Conclusiones insertada.")
