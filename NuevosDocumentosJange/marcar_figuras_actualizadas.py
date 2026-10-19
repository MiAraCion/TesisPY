# -*- coding: utf-8 -*-
"""Agrega una etiqueta de texto resaltada en amarillo justo debajo de cada
una de las 13 figuras que se reemplazaron, para poder identificarlas
visualmente (el resaltado de Word no se puede aplicar a una imagen en si)."""

import copy
from pathlib import Path

from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.text.paragraph import Paragraph
from docx.shared import Pt
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

DOC_PATH = Path(r"c:\Users\Remoto\Desktop\TesisPY\NuevosDocumentosJange\Plantilla de Tesis - ARAGON - JANGE - Final.docx")

doc = Document(DOC_PATH)

# mismos indices (4 a 16) que se reemplazaron en actualizar_figuras_indice.py
INDICES_ACTUALIZADOS = list(range(4, 17))

shapes = doc.inline_shapes
print(f"Total de figuras: {len(shapes)}")

etiquetados = 0
for idx in INDICES_ACTUALIZADOS:
    shape = shapes[idx]
    inline_elem = shape._inline
    # subir desde el <wp:inline> hasta el <w:p> (parrafo) que lo contiene
    run_elem = inline_elem.getparent().getparent()  # w:drawing -> w:r
    p_elem = run_elem.getparent()  # w:r -> w:p
    if p_elem.tag != qn('w:p'):
        print(f"   ADVERTENCIA: shape {idx} no esta dentro de un <w:p> como se esperaba, se omite")
        continue

    # crear un nuevo parrafo justo despues del parrafo de la imagen
    new_p_elem = OxmlElement('w:p')
    pPr = OxmlElement('w:pPr')
    jc = OxmlElement('w:jc')
    jc.set(qn('w:val'), 'center')
    pPr.append(jc)
    new_p_elem.append(pPr)
    p_elem.addnext(new_p_elem)

    new_para = Paragraph(new_p_elem, doc)
    run = new_para.add_run("[FIGURA ACTUALIZADA — datos recalculados tras fix C3/C4]")
    run.font.highlight_color = WD_COLOR_INDEX.YELLOW
    run.italic = True
    run.font.size = Pt(9)
    etiquetados += 1
    print(f"   shape {idx}: etiqueta agregada")

doc.save(DOC_PATH)
print(f"\n{etiquetados} figuras etiquetadas y guardadas.")
