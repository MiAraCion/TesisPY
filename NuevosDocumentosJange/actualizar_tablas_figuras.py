# -*- coding: utf-8 -*-
"""Actualiza las 8 tablas de datos, el texto narrativo con numeros desactualizados,
las 13 figuras de datos, y el indice de tablas/figuras, en el documento real de
tesis, resaltando en amarillo todo lo modificado."""

import copy
import shutil
from pathlib import Path

from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.text.run import Run
from docx.shared import Pt

DOC_PATH = Path(r"c:\Users\Remoto\Desktop\TesisPY\NuevosDocumentosJange\Plantilla de Tesis - ARAGON - JANGE - Final.docx")
PROJ = Path(r"c:\Users\Remoto\Desktop\TesisPY")

doc = Document(DOC_PATH)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def set_cell_highlighted(cell, text, bold=None):
    """Reemplaza el texto de una celda, conservando fuente/tamano del primer
    run existente cuando sea posible, y resalta en amarillo."""
    old_runs = cell.paragraphs[0].runs
    font_name, font_size, was_bold = None, None, bold
    if old_runs:
        r0 = old_runs[0]
        font_name = r0.font.name
        font_size = r0.font.size
        if was_bold is None:
            was_bold = r0.bold
    cell.text = str(text)
    run = cell.paragraphs[0].runs[0]
    if font_name:
        run.font.name = font_name
    if font_size:
        run.font.size = font_size
    else:
        run.font.size = Pt(8.5)
    if was_bold:
        run.bold = True
    run.font.highlight_color = WD_COLOR_INDEX.YELLOW


def highlight_replace_in_paragraph(paragraph, old_fragment, new_fragment):
    """Busca old_fragment dentro del texto COMPLETO del parrafo (puede
    abarcar varios runs), lo reemplaza por new_fragment, y resalta en
    amarillo solo el fragmento nuevo. Devuelve True si hizo el cambio."""
    full_text = "".join(r.text for r in paragraph.runs)
    idx = full_text.find(old_fragment)
    if idx == -1:
        return False
    before = full_text[:idx]
    after = full_text[idx + len(old_fragment):]

    # tomar formato del primer run como referencia
    ref_run = paragraph.runs[0]
    ref_elem = ref_run._element

    # limpiar todos los runs existentes del parrafo
    for r in list(paragraph.runs):
        r._element.getparent().remove(r._element)

    def add_run(text, highlight=False):
        new_elem = copy.deepcopy(ref_elem)
        # limpiar highlight previo si lo hubiera
        paragraph._p.append(new_elem)
        run = Run(new_elem, paragraph)
        run.text = text
        run.font.highlight_color = WD_COLOR_INDEX.YELLOW if highlight else None
        return run

    if before:
        add_run(before, highlight=False)
    add_run(new_fragment, highlight=True)
    if after:
        add_run(after, highlight=False)
    return True


cambios_texto = []


def fix_text(idx_parrafo, old_fragment, new_fragment, etiqueta):
    ok = highlight_replace_in_paragraph(doc.paragraphs[idx_parrafo], old_fragment, new_fragment)
    cambios_texto.append((idx_parrafo, etiqueta, ok))
    if not ok:
        print(f"   ADVERTENCIA: no se encontro el fragmento esperado en parrafo {idx_parrafo} ({etiqueta})")


# =================================================================
# PARTE 1: TABLAS
# =================================================================
print("Actualizando tablas...")

# --- Tabla indice 9: One-Hot / Escalado ---
t9 = doc.tables[9]
for row_idx, modelo in enumerate(["Random Forest", "Gradient Boosting", "XGBoost"], start=1):
    set_cell_highlighted(t9.rows[row_idx].cells[1], "Sí")

# --- Tabla indice 10: Tabla 2 (Boruta, top 10) ---
t10 = doc.tables[10]
boruta_rows = [
    ("1", "V201", "Sí (confirmada)", "0.156", "15.6", "14.57"),
    ("2", "V222", "Sí (confirmada)", "0.152", "30.8", "11.68"),
    ("3", "V040", "Sí (confirmada)", "0.140", "44.8", "8.23"),
    ("4", "V012", "Sí (confirmada)", "0.106", "55.3", "8.53"),
    ("5", "IMC", "Sí (confirmada)", "0.092", "64.6", "6.48"),
    ("6", "V190", "Sí (confirmada)", "0.071", "71.7", "6.78"),
    ("7", "V025", "No confirmada", "0.063", "78.0", "5.83"),
    ("8", "V106", "No confirmada", "0.046", "82.7", "10.40"),
    ("9", "V501", "No confirmada", "0.030", "85.6", "15.26"),
    ("10", "QS206", "No confirmada", "0.027", "88.3", "9.20"),
]
for row_idx, fila in enumerate(boruta_rows, start=1):
    for col_idx, val in enumerate(fila):
        set_cell_highlighted(t10.rows[row_idx].cells[col_idx], val)

# --- Tabla indice 11: train/test antes-despues SMOTE ---
t11 = doc.tables[11]
smote_dist_rows = [
    ("Train, antes de SMOTE-NC", "28,550 (91.2%)", "2,738 (8.8%)", "31,288"),
    ("Train, después de SMOTE-NC", "28,550 (50%)", "28,550 (50%)", "57,100"),
    ("Test (sin modificar)", "7,138 (91.3%)", "684 (8.7%)", "7,822"),
]
for row_idx, fila in enumerate(smote_dist_rows, start=1):
    for col_idx, val in enumerate(fila):
        set_cell_highlighted(t11.rows[row_idx].cells[col_idx], val)

# --- Tabla indice 12: SMOTE con/sin (RF fijo) ---
t12 = doc.tables[12]
smote_metrics_rows = [
    ("SIN SMOTE-NC", "0.9196", "0.5402", "1.0000", "0.0804", "0.1488", "0.6784", "0.2440"),
    ("CON SMOTE-NC", "0.6080", "0.6147", "0.1317", "0.6228", "0.2175", "0.6570", "0.2360"),
]
for row_idx, fila in enumerate(smote_metrics_rows, start=1):
    for col_idx, val in enumerate(fila):
        set_cell_highlighted(t12.rows[row_idx].cells[col_idx], val)

# --- Tabla indice 13: Tabla 5, k-fold ---
t13 = doc.tables[13]
kfold_rows = [
    ("1", "Gradient Boosting", "0.2114", "0.006", "0.5529", "0.1307", "0.6499", "0.0111"),
    ("2", "XGBoost", "0.2103", "0.0094", "0.5123", "0.1323", "0.6476", "0.0106"),
    ("3", "Random Forest", "0.2085", "0.0062", "0.5778", "0.1274", "0.6463", "0.0113"),
    ("4", "SVM", "0.2074", "0.0054", "0.5652", "0.1271", "0.6264", "0.0100"),
    ("5", "KNN", "0.1927", "0.0068", "0.4644", "0.1216", "0.6072", "0.0128"),
]
for row_idx, fila in enumerate(kfold_rows, start=1):
    for col_idx, val in enumerate(fila):
        set_cell_highlighted(t13.rows[row_idx].cells[col_idx], val)

# --- Tabla indice 14: Tabla 6, hiperparametros (+ columna nueva IC95% AUROC) ---
t14 = doc.tables[14]
# agregar columna nueva para el IC 95% (T1)
t14.add_column(t14.columns[0].width if t14.columns else None)
set_cell_highlighted(t14.rows[0].cells[4], "IC 95% AUROC (bootstrap)", bold=True)
hiperparam_rows = [
    ("Gradient Boosting", "{'n_estimators': 150, 'max_depth': 2, 'learning_rate': 0.05}", "0.5520", "0.6540", "[0.6336, 0.6749]"),
    ("Random Forest", "{'n_estimators': 300, 'min_samples_leaf': 1, 'max_depth': 5}", "0.5012", "0.6499", "[0.6283, 0.6710]"),
    ("XGBoost", "{'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.05}", "0.5097", "0.6469", "[0.6252, 0.6686]"),
    ("SVM", "{'gamma': 'auto', 'C': 1}", "0.4128", "0.6191", "[0.5979, 0.6408]"),
    ("KNN", "{'weights': 'uniform', 'n_neighbors': 11}", "0.6364", "0.5883", "[0.5658, 0.6101]"),
]
for row_idx, fila in enumerate(hiperparam_rows, start=1):
    for col_idx, val in enumerate(fila):
        set_cell_highlighted(t14.rows[row_idx].cells[col_idx], val)

# --- Tabla indice 15: Tabla 7, matriz de confusion (Gradient Boosting, k-fold) ---
t15 = doc.tables[15]
set_cell_highlighted(t15.rows[1].cells[1], "23,100 (VN)")
set_cell_highlighted(t15.rows[1].cells[2], "12,588 (FP)")
set_cell_highlighted(t15.rows[2].cells[1], "1,530 (FN)")
set_cell_highlighted(t15.rows[2].cells[2], "1,892 (VP)")

# --- Tabla indice 16: Tabla 8, SHAP global (V501 agregada en una sola fila) ---
t16 = doc.tables[16]
shap_rows = [
    ("1", "V025", "0.0662", "Sociodemografico", "24.6"),
    ("2", "V501", "0.0552", "Sociodemografico", "20.5"),
    ("3", "V106", "0.0373", "Sociodemografico", "13.8"),
    ("4", "QS206", "0.0331", "Salud", "12.3"),
    ("5", "V201", "0.0193", "Salud", "7.2"),
    ("6", "V222", "0.0184", "Salud", "6.8"),
    ("7", "V190", "0.0134", "Sociodemografico", "5.0"),
    ("8", "V012", "0.0118", "Sociodemografico", "4.4"),
    ("9", "V040", "0.0090", "Sociodemografico", "3.3"),
    ("10", "IMC", "0.0059", "Salud", "2.2"),
]
for row_idx, fila in enumerate(shap_rows, start=1):
    for col_idx, val in enumerate(fila):
        set_cell_highlighted(t16.rows[row_idx].cells[col_idx], val)

print("Tablas actualizadas.")

# =================================================================
# PARTE 2: TEXTO NARRATIVO CON NUMEROS DESACTUALIZADOS
# =================================================================
print("Actualizando texto narrativo...")

fix_text(395, "el dataset consolidado de 29,450 registros",
         "el dataset consolidado de 39,110 registros", "4.10.2 dataset consolidado N")

fix_text(398, "109 registros, un 0.37% de los registros totales",
         "148 registros, un 0.38% de los registros totales", "4.10.2 IMC faltantes N")

fix_text(407, "80% para entrenamiento (23,560 registros) y 20% para prueba (5,890 registros)",
         "80% para entrenamiento (31,288 registros) y 20% para prueba (7,822 registros)",
         "4.10.2 particion 80/20 N")

fix_text(411, "cuales de las 20 variables candidatas",
         "cuales de las 19 variables candidatas", "4.10.3 N candidatas")

fix_text(392,
         "26,814 casos sin pérdida (91.05%) y 2,636 casos con pérdida gestacional (8.95%)",
         "35,688 casos sin pérdida (91.25%) y 3,422 casos con pérdida gestacional (8.75%)",
         "4.10.1 distribucion final target")

fix_text(418,
         "Boruta confirmo estadísticamente solo 4 como relevantes, altitud conglomerada (V040), intervalo intergenésico (V222), índice de masa corporal (IMC), índice de riqueza (V190). Las 16 variables restantes",
         "Boruta confirmo estadísticamente 6 como relevantes: paridad (V201), intervalo intergenésico (V222), altitud conglomerada (V040), edad (V012), índice de masa corporal (IMC) e índice de riqueza (V190). Las 13 variables restantes",
         "4.10.3 boruta confirmadas 4->6")

fix_text(418,
         "habría dejado fuera predictores con fuerte respaldo teórico en la literatura de salud reproductiva [como la edad, el nivel educativo o la paridad], que en esta muestra no cruzaron el umbral estadístico de Boruta",
         "habría dejado fuera predictores con respaldo teórico en la literatura de salud reproductiva [como el nivel educativo, el área de residencia o el estado civil], que en esta muestra no cruzaron el umbral estadístico de Boruta",
         "4.10.3 ejemplos variables no confirmadas")

fix_text(418, "que en conjunto explican el 86.1% de la importancia total",
         "que en conjunto explican el 88.3% de la importancia total", "4.10.3 acumulado top10")

fix_text(426,
         "al reentrenar los 5 modelos usando únicamente las 4 variables confirmadas por Boruta, los 5 obtuvieron un AUROC y F1 inferiores a los alcanzados con el criterio híbrido de 10 variables (ver Tabla comparativa, Anexo/4.10.6), confirmando que el respaldo teórico adicional aporta capacidad predictiva real y no solo ruido.",
         "al reentrenar los 5 modelos usando únicamente las 6 variables confirmadas por Boruta, se observó un desempeño comparable al del criterio híbrido de 10 variables (4 de 5 modelos con F1 similar o superior usando solo las 6 confirmadas; ver Tabla comparativa, Anexo/4.10.6). Se optó por mantener el criterio híbrido de 10 variables como conjunto oficial del modelamiento por su mayor respaldo teórico, documentando este hallazgo como referencia para investigaciones futuras.",
         "4.10.3 C1 resultado corregido")

fix_text(443,
         "Con SMOTE-NC, la Exactitud baja a 61.7%, pero el Recall sube a 48.2% y el F1-Score pasa de 0 a 0.18",
         "Con SMOTE-NC, la Exactitud baja a 60.8%, pero el Recall sube a 62.3% y el F1-Score pasa de 0.15 a 0.22",
         "4.10.4 SMOTE con/sin narrativa")

fix_text(486,
         "De los 2,636 casos reales de pérdida gestacional en el conjunto evaluado, el modelo detectó correctamente 1,284 (Recall ≈ 48.7%) y no detectó 1,352 (51.3%)",
         "De los 3,422 casos reales de pérdida gestacional en el conjunto evaluado, el modelo detectó correctamente 1,892 (Recall ≈ 55.3%) y no detectó 1,530 (44.7%)",
         "4.10.6 confusion matrix narrativa")

fix_text(486, "estos 1,352 falsos negativos representan", "estos 1,530 falsos negativos representan", "4.10.6 FN narrativa")

fix_text(486, "el AUC-ROC de los cinco modelos se mantiene en un rango moderado (entre 0.58 y 0.61 aproximadamente)",
         "el AUC-ROC de los cinco modelos se mantiene en un rango moderado (entre 0.59 y 0.65 aproximadamente)",
         "4.10.6 rango AUC narrativa")

fix_text(400,
         "Esta codificación solo se aplicó a los modelos de árboles que tampoco son inmunes a este problema, KNN y SVM. Los modelos de ensamble basados en árboles Random Forest, Gradient Boosting, XGBoost trabajan haciendo cortes sobre cada variable por separado, así que no se ven afectados de la misma forma por usar la codificación numérica original.",
         "A diferencia de lo que podría asumirse, los modelos de árboles (Random Forest, Gradient Boosting y XGBoost en su configuración estándar) tampoco son inmunes a este problema: al dividir mediante cortes de umbral sobre el valor numérico de la variable (p. ej., V501 ≤ 2), un código nominal sin orden real puede introducir agrupaciones artificiales entre categorías sin relación semántica entre sí. Por esta razón, el One-Hot Encoding de V501 se aplicó de manera uniforme a los cinco modelos, y no únicamente a KNN y SVM.",
         "4.10.2 One-Hot V501 correccion")

fix_text(500,
         "La variable con mayor peso resultó ser V025 (área de residencia, urbano/rural), que por sí sola concentra el 24.2% de la importancia total, seguida de V201 (paridad) y V501 (estado civil). Agrupando las 10 variables por dimensión, las sociodemográficas concentran el 62% de la importancia total del modelo, frente a un 38% de las variables de salud.",
         "La variable con mayor peso resultó ser V025 (área de residencia, urbano/rural), que por sí sola concentra el 24.6% de la importancia total, seguida de V501 (estado civil, 20.5%) y V106 (nivel educativo, 13.8%). Agrupando las 10 variables por dimensión, las sociodemográficas concentran el 71.5% de la importancia total del modelo, frente a un 28.5% de las variables de salud.",
         "4.10.7 SHAP narrativa")

fix_text(184, "no forman parte del fenómeno estudiado. Esta condición",
         "Esta condición", "3.1.1 contradiccion involuntaria")

print("Texto narrativo actualizado.")
for idx, etiqueta, ok in cambios_texto:
    estado = "OK" if ok else "NO ENCONTRADO"
    print(f"   [{estado}] parrafo {idx}: {etiqueta}")

doc.save(DOC_PATH)
print("\nGuardado (parte 1: tablas + texto).")
