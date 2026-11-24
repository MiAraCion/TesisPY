# -*- coding: utf-8 -*-
# Entrena los 5 modelos con sus hiperparametros REALES (Modelos/resultados_
# modelos_optimizado.csv) y los guarda como pipelines de scikit-learn
# completos (preprocesamiento + modelo) con joblib, para que la app de
# Python (app.py) los cargue y use directamente, sin reimplementar nada
# a mano (a diferencia de la version web, que si necesito reescribir en
# JavaScript).
#
# SVM: entrenar sobre las 57,100 filas completas con probability=True tarda
# ~9 horas (comprobado). Aqui se usa el MISMO truco que ya usa el pipeline
# oficial para buscar hiperparametros de SVM (ver Modelos/entrenamiento_
# modelos_optimizado.py, LIMITE_SVM_BUSQUEDA=8000): una submuestra
# estratificada de 8,000 filas del train balanceado. El resultado es un SVM
# real y genuino, pero NO es bit-a-bit el mismo modelo reportado en la
# tesis (ese se entreno con las 57,100 filas) -- se marca explicitamente
# como aproximado en los metadatos y en la interfaz.

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

BASE = Path(r"C:\Users\Remoto\Desktop\TesisPY")
OUT_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 2023
LIMITE_SVM = 8000  # misma submuestra que usa el pipeline oficial para SVM

train = pd.read_csv(BASE / "SMOTENC" / "train_balanceado.csv", encoding="utf-8-sig")
test = pd.read_csv(BASE / "SMOTENC" / "test_original.csv", encoding="utf-8-sig")

FEATURES = [c for c in train.columns if c != "target"]
X_train = train[FEATURES].copy()
y_train = train["target"].astype(int).copy()
X_test = test[FEATURES].copy()
y_test = test["target"].astype(int).copy()

COLUMNAS_NUMERICAS = ["V201", "V222", "V040", "V012", "IMC", "V190", "V025", "V106", "QS206"]

# ColumnTransformer 1: One-Hot de V501 (nominal), el resto pasa igual.
# ColumnTransformer 2 (solo KNN/SVM): escala las numericas, las dummies de
# V501 pasan sin escalar (igual que en el pipeline oficial).
onehot_v501 = ColumnTransformer(
    transformers=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["V501"])],
    remainder="passthrough",
    verbose_feature_names_out=False,
).set_output(transform="pandas")

escalado_numericas = ColumnTransformer(
    transformers=[("escalado", StandardScaler(), COLUMNAS_NUMERICAS)],
    remainder="passthrough",
    verbose_feature_names_out=False,
).set_output(transform="pandas")

HIP = {
    "random_forest": dict(n_estimators=300, min_samples_leaf=1, max_depth=5),
    "gradient_boosting": dict(n_estimators=150, max_depth=2, learning_rate=0.05),
    "xgboost": dict(n_estimators=300, max_depth=3, learning_rate=0.05),
    "knn": dict(weights="uniform", n_neighbors=11),
    "svm": dict(gamma="auto", C=1),
}
UMBRAL = {
    "random_forest": 0.5012,
    "gradient_boosting": 0.5520,
    "xgboost": 0.5097000002861023,
    "knn": 0.6364,
    "svm": 0.4128,
}
NOMBRES = {
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost",
    "knn": "KNN",
    "svm": "SVM (aproximado, submuestra 8,000 filas)",
}

resultados_verif = {}


def evaluar_y_guardar(clave, pipeline, n_filas_entrenamiento, nota=""):
    t0 = time.time()
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= UMBRAL[clave]).astype(int)
    auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, pred)
    recall = recall_score(y_test, pred)
    precision = precision_score(y_test, pred)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print(f"\n{NOMBRES[clave]}")
    print(f"  Filas de entrenamiento usadas: {n_filas_entrenamiento:,}")
    print(f"  AUROC={auc:.4f}  F1={f1:.4f}  Recall={recall:.4f}  Precision={precision:.4f}")
    print(f"  TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"  Tiempo de evaluacion sobre test: {time.time()-t0:.1f}s")

    resultados_verif[clave] = dict(
        nombre=NOMBRES[clave], umbral=UMBRAL[clave], n_filas_entrenamiento=int(n_filas_entrenamiento),
        auc=round(float(auc), 4), f1=round(float(f1), 4), recall=round(float(recall), 4),
        precision=round(float(precision), 4), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), nota=nota,
    )
    ruta = OUT_DIR / f"modelo_{clave}.pkl"
    joblib.dump(pipeline, ruta)
    print(f"  Guardado: {ruta} ({ruta.stat().st_size/1024:,.1f} KB)")


# ======================================================================
# Random Forest / Gradient Boosting / XGBoost -- sin escalado, full train
# ======================================================================
print("=" * 60)
print("Random Forest")
print("=" * 60)
pipe_rf = Pipeline([
    ("onehot", onehot_v501),
    ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **HIP["random_forest"])),
])
pipe_rf.fit(X_train, y_train)
evaluar_y_guardar("random_forest", pipe_rf, len(X_train))

print("\n" + "=" * 60)
print("Gradient Boosting")
print("=" * 60)
pipe_gb = Pipeline([
    ("onehot", onehot_v501),
    ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE, **HIP["gradient_boosting"])),
])
pipe_gb.fit(X_train, y_train)
evaluar_y_guardar("gradient_boosting", pipe_gb, len(X_train))

print("\n" + "=" * 60)
print("XGBoost")
print("=" * 60)
pipe_xgb = Pipeline([
    ("onehot", onehot_v501),
    ("clf", XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1, **HIP["xgboost"])),
])
pipe_xgb.fit(X_train, y_train)
evaluar_y_guardar("xgboost", pipe_xgb, len(X_train))

# ======================================================================
# KNN -- con escalado, full train (rapido, sin problema)
# ======================================================================
print("\n" + "=" * 60)
print("KNN")
print("=" * 60)
pipe_knn = Pipeline([
    ("onehot", onehot_v501),
    ("escalado", escalado_numericas),
    ("clf", KNeighborsClassifier(**HIP["knn"])),
])
pipe_knn.fit(X_train, y_train)
evaluar_y_guardar("knn", pipe_knn, len(X_train))

# ======================================================================
# SVM -- con escalado, SUBMUESTRA de 8,000 filas (igual que la busqueda de
# hiperparametros del pipeline oficial), para que sea viable en minutos.
# ======================================================================
print("\n" + "=" * 60)
print(f"SVM (submuestra de {LIMITE_SVM:,} filas, igual que la busqueda oficial)")
print("=" * 60)
X_train_svm, _, y_train_svm, _ = train_test_split(
    X_train, y_train, train_size=LIMITE_SVM, random_state=RANDOM_STATE, stratify=y_train,
)
pipe_svm = Pipeline([
    ("onehot", onehot_v501),
    ("escalado", escalado_numericas),
    ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE, **HIP["svm"])),
])
t0 = time.time()
pipe_svm.fit(X_train_svm, y_train_svm)
print(f"Tiempo de entrenamiento SVM: {time.time()-t0:.1f}s")
n_sv = pipe_svm.named_steps["clf"].support_vectors_.shape[0]
print(f"Vectores de soporte: {n_sv:,} de {len(X_train_svm):,} filas de entrenamiento")
evaluar_y_guardar(
    "svm", pipe_svm, len(X_train_svm),
    nota=(f"APROXIMADO: entrenado sobre una submuestra de {LIMITE_SVM:,} filas (no las 57,100 "
          f"completas) por costo computacional -- el SVM oficial de la tesis usa el set completo "
          f"y tarda ~9h en entrenar. {n_sv:,} vectores de soporte."),
)

# ======================================================================
# Metadatos comunes para la app
# ======================================================================
metadatos = {
    "features_originales": FEATURES,
    "hiperparametros": HIP,
    "umbrales": UMBRAL,
    "nombres": NOMBRES,
    "resultados_verificacion": resultados_verif,
}
with open(OUT_DIR / "metadatos.json", "w", encoding="utf-8") as f:
    json.dump(metadatos, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print("EXPORTACION COMPLETA -- 5 pipelines guardados en", OUT_DIR)
print("=" * 60)
