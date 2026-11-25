# -*- coding: utf-8 -*-
# Reentrena el SVM con el 80% COMPLETO de entrenamiento (57,100 filas
# balanceadas), igual que el modelo oficial reportado en la tesis -- NO la
# submuestra de 8,000 filas usada antes como aproximacion rapida.
# ADVERTENCIA: esto tarda ~9 horas (comprobado empiricamente antes en esta
# misma maquina, con probability=True sobre 57,100 filas, kernel RBF).
#
# Al terminar: sobreescribe DemoPrediccion/modelo_svm.pkl (la version
# aproximada anterior) y actualiza la entrada "svm" en metadatos.json para
# que la app ya no muestre la advertencia de "aproximado".

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

BASE = Path(r"C:\Users\Remoto\Desktop\TesisPY")
OUT_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 2023
UMBRAL_SVM = 0.4128

train = pd.read_csv(BASE / "SMOTENC" / "train_balanceado.csv", encoding="utf-8-sig")
test = pd.read_csv(BASE / "SMOTENC" / "test_original.csv", encoding="utf-8-sig")

FEATURES = [c for c in train.columns if c != "target"]
X_train = train[FEATURES].copy()
y_train = train["target"].astype(int).copy()
X_test = test[FEATURES].copy()
y_test = test["target"].astype(int).copy()

COLUMNAS_NUMERICAS = ["V201", "V222", "V040", "V012", "IMC", "V190", "V025", "V106", "QS206"]

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

print(f"Entrenando SVM sobre el 80% COMPLETO de entrenamiento: {len(X_train):,} filas")
print("Hiperparametros: kernel=rbf, gamma=auto, C=1, probability=True, random_state=2023")
print("Esto puede tardar varias horas. Inicio:", time.strftime("%Y-%m-%d %H:%M:%S"))

pipe_svm = Pipeline([
    ("onehot", onehot_v501),
    ("escalado", escalado_numericas),
    ("clf", SVC(kernel="rbf", probability=True, gamma="auto", C=1, random_state=RANDOM_STATE)),
])

t0 = time.time()
pipe_svm.fit(X_train, y_train)
dt = time.time() - t0
print(f"\nEntrenamiento completado en {dt:.1f}s ({dt/3600:.2f} horas). Fin:", time.strftime("%Y-%m-%d %H:%M:%S"))

n_sv = pipe_svm.named_steps["clf"].support_vectors_.shape[0]
print(f"Vectores de soporte: {n_sv:,} de {len(X_train):,} filas de entrenamiento")

proba = pipe_svm.predict_proba(X_test)[:, 1]
pred = (proba >= UMBRAL_SVM).astype(int)
auc = roc_auc_score(y_test, proba)
f1 = f1_score(y_test, pred)
recall = recall_score(y_test, pred)
precision = precision_score(y_test, pred)
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
print(f"\nAUROC={auc:.4f}  F1={f1:.4f}  Recall={recall:.4f}  Precision={precision:.4f}")
print(f"TN={tn} FP={fp} FN={fn} TP={tp}")

ruta_pkl = OUT_DIR / "modelo_svm.pkl"
joblib.dump(pipe_svm, ruta_pkl)
print(f"\nGuardado: {ruta_pkl} ({ruta_pkl.stat().st_size/1024:,.1f} KB)")

# Actualizar metadatos.json -- solo la entrada "svm"
ruta_meta = OUT_DIR / "metadatos.json"
with open(ruta_meta, encoding="utf-8") as f:
    metadatos = json.load(f)

metadatos["resultados_verificacion"]["svm"] = dict(
    nombre="SVM",
    umbral=UMBRAL_SVM,
    n_filas_entrenamiento=int(len(X_train)),
    auc=round(float(auc), 4), f1=round(float(f1), 4), recall=round(float(recall), 4),
    precision=round(float(precision), 4), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    nota=(f"Entrenado con el 80% completo de entrenamiento ({len(X_train):,} filas), igual que el "
          f"modelo oficial reportado en la tesis. {n_sv:,} vectores de soporte. "
          f"Tiempo de entrenamiento: {dt/3600:.1f} horas."),
)
metadatos["nombres"]["svm"] = "SVM"

with open(ruta_meta, "w", encoding="utf-8") as f:
    json.dump(metadatos, f, ensure_ascii=False, indent=2)
print(f"Actualizado: {ruta_meta}")

print("\nSVM COMPLETO (80% de entrenamiento) LISTO")
