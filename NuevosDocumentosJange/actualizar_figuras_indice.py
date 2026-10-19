# -*- coding: utf-8 -*-
"""Parte 2: reemplaza las 13 figuras de datos por las versiones regeneradas
tras C3/C4, y reconstruye el indice de tablas y figuras con los titulos
reales tal como aparecen en el cuerpo del documento."""

import copy
from pathlib import Path

from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.text.run import Run
from docx.shared import Pt
from docx.oxml.ns import qn

DOC_PATH = Path(r"c:\Users\Remoto\Desktop\TesisPY\NuevosDocumentosJange\Plantilla de Tesis - ARAGON - JANGE - Final.docx")
PROJ = Path(r"c:\Users\Remoto\Desktop\TesisPY")

doc = Document(DOC_PATH)

# =================================================================
# PARTE 2A: REEMPLAZAR IMAGENES (shapes 4 a 16, ver mapeo verificado)
# =================================================================
print("Reemplazando figuras de datos...")

MAPEO_IMAGENES = {
    4:  PROJ / "boruta" / "grafico_importancia.png",
    5:  PROJ / "Validacion_SMOTE" / "grafico_con_sin_smote.png",
    6:  PROJ / "Modelos" / "grafico_kfold.png",
    7:  PROJ / "Modelos" / "grafico_comparacion_optimizado.png",
    8:  PROJ / "Modelos" / "matriz_confusion_mejor_modelo.png",
    9:  PROJ / "XAI_Global" / "shap_beeswarm.png",
    10: PROJ / "XAI_Global" / "shap_barras_importancia.png",
    11: PROJ / "XAI_Local" / "lime_alto_riesgo.png",
    12: PROJ / "XAI_Local" / "shap_waterfall_alto_riesgo.png",
    13: PROJ / "XAI_Local" / "lime_bajo_riesgo.png",
    14: PROJ / "XAI_Local" / "shap_waterfall_bajo_riesgo.png",
    15: PROJ / "XAI_Local" / "lime_caso_limite.png",
    16: PROJ / "XAI_Local" / "shap_waterfall_caso_limite.png",
}

shapes = doc.inline_shapes
print(f"Total de figuras en el documento: {len(shapes)}")

for idx, ruta_png in MAPEO_IMAGENES.items():
    if not ruta_png.exists():
        print(f"   ADVERTENCIA: no existe {ruta_png}, se omite shape {idx}")
        continue
    shape = shapes[idx]
    blip = shape._inline.graphic.graphicData.pic.blipFill.blip
    rId = blip.get(qn('r:embed'))
    image_part = doc.part.related_parts[rId]
    with open(ruta_png, 'rb') as f:
        image_part._blob = f.read()
    print(f"   shape {idx} -> {ruta_png.name} (reemplazada)")

print("Figuras reemplazadas (shapes 4-16). shape 3 (parrafo 408) no se toco: no se pudo identificar con certeza a que figura corresponde, revisar manualmente.")

# =================================================================
# PARTE 2B: RECONSTRUIR INDICE DE FIGURAS Y DE TABLAS
# =================================================================
print("\nReconstruyendo indice de tablas y figuras...")

FIGURAS = [
    "Figura 1: Elaboración de Target",
    "Figura 2: Definición de Muestra",
    "Figura 3: Descripción de las fases",
    "Figura 2. Importancia de las 19 variables candidatas según Boruta, ordenadas de mayor a menor. (sección 4.10.3)",
    "Figura 3. Comparación de métricas del modelo sin SMOTE-NC frente a con SMOTE-NC. (sección 4.10.4)",
    "Figura 4. F1-Score y AUC-ROC promedio (con barras de error) de los 5 modelos, validación cruzada 5-fold. (sección 4.10.5)",
    "Figura 5. Comparación de los 5 modelos ya optimizados, evaluados con su umbral óptimo respectivo. (sección 4.10.5)",
    "Figura 6. Matriz de confusión acumulada del mejor modelo (Gradient Boosting), validación cruzada 5-fold. (sección 4.10.6)",
    "Figura 7. Gráfico SHAP Beeswarm: distribución del impacto de cada variable sobre la predicción. (sección 4.10.7)",
    "Figura 8. Importancia global de cada variable según el valor absoluto medio de SHAP. (sección 4.10.7)",
    "Figura 9. Explicación LIME del caso de alto riesgo (verdadero positivo). (sección 4.10.7)",
    "Figura 10. Gráfico SHAP Waterfall del caso de alto riesgo (verdadero positivo). (sección 4.10.7)",
    "Figura 11. Explicación LIME del caso de bajo riesgo (verdadero negativo). (sección 4.10.7)",
    "Figura 12. Gráfico SHAP Waterfall del caso de bajo riesgo (verdadero negativo). (sección 4.10.7)",
    "Figura 13. Explicación LIME del caso límite (falso positivo, probabilidad ≈ 0.5). (sección 4.10.7)",
    "Figura 14. Gráfico SHAP Waterfall del caso límite (falso positivo, probabilidad ≈ 0.5). (sección 4.10.7)",
]

TABLAS = [
    "Tabla 1: Métricas de modelos de Machine Learning en Salud Reproductiva (sección 2.3)",
    "Tabla 2: Variables Determinantes Sociodemográficos (sección 3.2.1)",
    "Tabla 3: Variables de Determinantes de Salud (sección 3.2.2)",
    "Tabla 4: Comparación de tipos de modelos de Machine Learning (sección 3.3.2)",
    "Tabla 5: Niveles de Análisis de Explicabilidad Global (sección 3.3.5)",
    "Tabla 6: Niveles de Análisis de Explicabilidad Local (sección 3.3.5)",
    "Tabla 7: Integración de Bases de Datos de la ENDES (sección 4.5)",
    "Tabla 8: Matriz de Confusión (sección 4.8)",
    "Tabla 2. Las 10 variables retenidas para el modelado, de un total de 19 evaluadas por Boruta. (sección 4.10.3)",
    "Tabla 3. Distribución de clases en train antes/después de SMOTE-NC, y en test. (sección 4.10.4)",
    "Tabla 4. Comparación de un mismo modelo (Random Forest) entrenado sin y con SMOTE-NC. (sección 4.10.4)",
    "Tabla 5. Resultados promedio (± desviación estándar) de la validación cruzada de 5 particiones. (sección 4.10.5)",
    "Tabla 6. Hiperparámetros, umbral de decisión e intervalo de confianza del AUROC óptimos para cada modelo. (sección 4.10.5)",
    "Tabla 7. Matriz de confusión acumulada (5-fold) del modelo Gradient Boosting. (sección 4.10.6)",
    "Tabla 8. Importancia global de las 10 variables del modelo, según el valor absoluto medio de SHAP. (sección 4.10.7)",
]


def find_heading(texto_buscado):
    for i, para in enumerate(doc.paragraphs):
        if para.text.strip() == texto_buscado:
            return i
    return None


def limpiar_parrafos_entre(inicio, fin):
    """Elimina todos los parrafos entre inicio (exclusivo) y fin (exclusivo)."""
    parrafos = doc.paragraphs
    for i in range(fin - 1, inicio, -1):
        p = parrafos[i]._p
        p.getparent().remove(p)


def insertar_lista(despues_de_parrafo, items):
    """Inserta una lista de parrafos (uno por item) justo despues del
    parrafo dado, resaltados en amarillo, devuelve el ultimo parrafo insertado."""
    ref = despues_de_parrafo._p
    ultimo = despues_de_parrafo
    for item in items:
        new_p = copy.deepcopy(ref)
        for r in list(new_p.findall(qn('w:r'))):
            new_p.remove(r)
        ref.addnext(new_p)
        ref = new_p
        from docx.text.paragraph import Paragraph
        p_obj = Paragraph(new_p, despues_de_parrafo._parent)
        run = p_obj.add_run(item)
        run.font.highlight_color = WD_COLOR_INDEX.YELLOW
        run.font.size = Pt(10.5)
        ultimo = p_obj
    return ultimo


# --- Indice de Tablas: se procesa primero (esta mas abajo en el documento) ---
idx_tablas = find_heading("Índice de Tablas")
# encontrar el rango de parrafos viejos del indice de tablas (hasta el siguiente parrafo no vacio con salto de seccion)
# los parrafos viejos conocidos: 46-49 (pueden variar si el documento cambio arriba)
fin_tablas = idx_tablas + 1
while fin_tablas < len(doc.paragraphs) and doc.paragraphs[fin_tablas].text.strip().startswith("Tabla"):
    fin_tablas += 1
limpiar_parrafos_entre(idx_tablas, fin_tablas)
insertar_lista(doc.paragraphs[idx_tablas], TABLAS)

# --- Indice de Figuras ---
idx_figuras = find_heading("Índice de figuras")
fin_figuras = idx_figuras + 1
while fin_figuras < len(doc.paragraphs) and doc.paragraphs[fin_figuras].text.strip().startswith("Figura"):
    fin_figuras += 1
limpiar_parrafos_entre(idx_figuras, fin_figuras)
insertar_lista(doc.paragraphs[idx_figuras], FIGURAS)

print("Indice de tablas y figuras reconstruido (resaltado en amarillo).")
print("NOTA: no se incluyeron numeros de pagina reales (no calculables sin abrir Word);")
print("      hay que actualizar la paginacion en Word antes de entregar (clic derecho > actualizar campo,")
print("      o revisar manualmente cada numero de pagina).")

doc.save(DOC_PATH)
print("\nGuardado (parte 2: figuras + indice).")
