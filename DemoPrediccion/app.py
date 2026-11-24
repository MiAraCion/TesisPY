# -*- coding: utf-8 -*-
# App de escritorio (Streamlit) para la demo en vivo, version Python.
# A diferencia de la interfaz web (HTML/JS), aqui los 5 modelos son los
# objetos reales de scikit-learn/XGBoost, cargados con joblib y usados con
# predict_proba() tal cual -- sin reimplementar ninguna formula a mano.
#
# Para correrla:  streamlit run app.py   (desde esta carpeta)
import json
import time
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Demo en vivo (Python)", layout="wide")

MAP_AREA = {"Urbano": 1, "Rural": 2}
MAP_EDUC = {"Sin educación": 0, "Primaria": 1, "Secundaria": 2, "Superior": 3}
MAP_RIQUEZA = {"Más pobre": 1, "Pobre": 2, "Medio": 3, "Rico": 4, "Más rico": 5}
MAP_CIVIL = {"Nunca unida": 0, "Casada": 1, "Conviviente": 2, "Viuda": 3, "Divorciada": 4, "Separada": 5}
MAP_ALCOHOL = {"Sí": 1, "No": 2}


@st.cache_resource
def cargar_metadatos():
    with open(BASE_DIR / "metadatos.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def cargar_modelo(clave):
    return joblib.load(BASE_DIR / f"modelo_{clave}.pkl")


def fila_a_features(fila, columnas_originales):
    # Reconstruye el vector de entrada EXACTAMENTE como lo espera el pipeline
    # entrenado. El target de la fila nunca entra aqui.
    valores = {
        "V201": fila["paridad"], "V222": fila["intervalo_intergenesico"], "V040": fila["altitud"],
        "V012": fila["edad"], "IMC": fila["imc"], "V190": MAP_RIQUEZA[fila["riqueza"]],
        "V025": MAP_AREA[fila["area"]], "V106": MAP_EDUC[fila["nivel_educativo"]],
        "QS206": MAP_ALCOHOL[fila["alcohol"]], "V501": MAP_CIVIL[fila["estado_civil"]],
    }
    return pd.DataFrame([[valores[c] for c in columnas_originales]], columns=columnas_originales)


metadatos = cargar_metadatos()
CLAVES = ["gradient_boosting", "random_forest", "xgboost", "knn", "svm"]

st.title("Demo en vivo del modelo predictivo — versión Python")
st.caption(
    "Cada predicción se calcula en tiempo real con el pipeline de scikit-learn / XGBoost "
    "realmente entrenado (joblib.load + predict_proba), sin reimplementar nada a mano. "
    "El target real nunca se entrega al modelo — solo se usa después del cálculo, para "
    "marcar Acierto / No acierto."
)

archivo = st.file_uploader("Cargar CSV de datos de prueba", type="csv")
ruta_default = BASE_DIR / "datos_prueba_20porciento.csv"

if archivo is not None:
    df = pd.read_csv(archivo, encoding="utf-8-sig")
elif ruta_default.exists():
    st.info(f"Usando por defecto **{ruta_default.name}** (puedes subir otro CSV arriba).")
    df = pd.read_csv(ruta_default, encoding="utf-8-sig")
else:
    df = None

if df is not None:
    c1, c2, c3 = st.columns(3)
    with c1:
        clave = st.selectbox("Algoritmo", CLAVES, format_func=lambda c: metadatos["nombres"][c])
    with c2:
        velocidad = st.select_slider("Velocidad", options=["Lento", "Normal", "Rápido"], value="Normal")
    with c3:
        n_filas = st.number_input(
            "Cuántos casos correr", min_value=10, max_value=int(len(df)),
            value=min(200, len(df)), step=10,
        )

    info_modelo = metadatos["resultados_verificacion"][clave]
    if info_modelo.get("nota"):
        st.warning(info_modelo["nota"])
    st.caption(
        f"Hiperparámetros reales: {metadatos['hiperparametros'][clave]} · "
        f"Umbral de decisión: {metadatos['umbrales'][clave]} · "
        f"Entrenado con {info_modelo['n_filas_entrenamiento']:,} filas · "
        f"AUROC(test)={info_modelo['auc']} · F1(test)={info_modelo['f1']}"
    )

    pausa = {"Lento": 0.9, "Normal": 0.4, "Rápido": 0.05}[velocidad]

    if st.button("▶ Correr demo", type="primary"):
        modelo = cargar_modelo(clave)
        umbral = metadatos["umbrales"][clave]
        columnas_originales = metadatos["features_originales"]

        barra = st.progress(0)
        s1, s2, s3, s4 = st.columns(4)
        m_proc, m_ok, m_fail, m_acc = s1.empty(), s2.empty(), s3.empty(), s4.empty()
        tabla = st.empty()

        aciertos = 0
        filas_resultado = []
        total = int(n_filas)
        for i in range(total):
            fila = df.iloc[i]
            x = fila_a_features(fila, columnas_originales)
            proba = float(modelo.predict_proba(x)[0, 1])
            pred = int(proba >= umbral)
            real = int(fila["target"])
            acierto = pred == real
            aciertos += int(acierto)

            filas_resultado.append({
                "#": int(fila["caso"]), "Edad": int(fila["edad"]), "Área": fila["area"],
                "Nivel educ.": fila["nivel_educativo"], "Riqueza": fila["riqueza"],
                "Estado civil": fila["estado_civil"], "IMC": fila["imc"],
                "Target real": "Con pérdida" if real == 1 else "Sin pérdida",
                "Predicción": "Con pérdida" if pred == 1 else "Sin pérdida",
                "Probabilidad": f"{proba*100:.1f}%",
                "Resultado": "✅ Acierto" if acierto else "❌ No acierto",
            })

            m_proc.metric("Procesados", f"{i+1}/{total}")
            m_ok.metric("Aciertos", aciertos)
            m_fail.metric("Fallos", (i + 1) - aciertos)
            m_acc.metric("Accuracy acumulado", f"{aciertos/(i+1)*100:.1f}%")
            barra.progress((i + 1) / total)
            tabla.dataframe(pd.DataFrame(filas_resultado[::-1]), use_container_width=True, height=440)
            time.sleep(pausa)

        st.success(f"Demo completada: {total} casos procesados con {metadatos['nombres'][clave]}.")
else:
    st.info("Coloca o sube el CSV de datos de prueba (datos_prueba_20porciento.csv) para comenzar.")
