# Resultados y Discusión — Pipeline ML/XAI (Fases 1-5)

*Nota: este documento está redactado en formato de sección de tesis, listo para pegar en el capítulo de resultados/discusión. Las citas usan formato (Autor, Año); ajústalas a la numeración de tu lista de referencias existente. Las fuentes completas están en `Fuentes/bibliografia_fuentes.md`.*

---

## Fase 1: Selección de características (Boruta)

La aplicación del algoritmo Boruta sobre las 20 variables candidatas (factores sociodemográficos y de salud, según la matriz de operacionalización) confirmó estadísticamente **4 variables**: altitud del conglomerado (V040), intervalo intergenésico (V222), índice de masa corporal (IMC) e índice de riqueza (V190), con importancia acumulada del 61.4% sobre el total evaluado. Ninguna variable quedó en estado "tentativo" en ninguna de las corridas del proyecto, lo que indica que Boruta alcanzó un veredicto definitivo (confirmada o rechazada) para cada variable, sin ambigüedad estadística residual.

Como indicador complementario de estabilidad, se calculó un Z-Score aproximado para cada variable (media/desviación estándar de su importancia a lo largo de las iteraciones válidas del algoritmo). Cabe precisar una limitación técnica: la implementación de Boruta utilizada (`boruta_py`, el puerto a Python del algoritmo original en R) no expone las importancias de las variables "sombra" de forma individual, por lo que no fue posible calcular el Z-Score exacto "variable real vs. mejor sombra" que define el algoritmo original. El estadístico aquí reportado es una aproximación de estabilidad, no el Z-Score canónico de Boruta — limitación que debe declararse explícitamente en la sección de limitaciones metodológicas.

Las variables confirmadas presentaron los Z-Score de estabilidad más bajos del conjunto (4.98–10.87), mientras que variables rechazadas como el estado civil (V501, Z=31.66) o la edad (V012, Z=29.14) mostraron valores de estabilidad más altos pero sin traducirse en una importancia mayor — es decir, su comportamiento fue consistente entre iteraciones, pero consistentemente bajo, no alto.

**HE1** ("Boruta selecciona un subconjunto estadísticamente significativo... reduciendo la dimensionalidad y eliminando el ruido sin pérdida de capacidad predictiva") se considera **parcialmente confirmada**: Boruta sí redujo la dimensionalidad de 20 a 4 variables con evidencia estadística, pero la comparación de capacidad predictiva (ver Fase 3) muestra que el conjunto ampliado top-10 y el conjunto de 4 confirmadas rinden de forma similar, sin una pérdida ni una ganancia sustancial — coherente con la naturaleza conservadora de Boruta, que prioriza evitar falsos positivos sobre maximizar el poder predictivo bruto.

---

## Fase 2: Tratamiento del desbalance (SMOTE-NC)

Se realizó una comparación controlada: el mismo modelo (Random Forest, mismos hiperparámetros y semilla) evaluado sobre el mismo conjunto de prueba, entrenado una vez sobre el conjunto de entrenamiento desbalanceado (~91%/9%, "sin procesar") y otra vez sobre el conjunto balanceado mediante SMOTE-NC (Chawla et al., 2002) al 50%/50%.

| Condición | Accuracy | Recall (clase 1) | F1 (clase 1) | AUC-ROC | PR-AUC |
|---|---|---|---|---|---|
| Sin SMOTE-NC | 0.9105 | **0.0000** | **0.0000** | 0.6176 | 0.1265 |
| Con SMOTE-NC | 0.6179 | **0.4741** | **0.1818** | 0.5940 | 0.1145 |

El hallazgo central es contundente: sin balanceo, el modelo predice la totalidad de los casos como "sin pérdida" (Recall = Precisión = F1 = 0), a pesar de exhibir una exactitud aparente del 91.05% — un ejemplo textual del fenómeno de "paradoja de la exactitud" bien documentado en clasificación con clases desbalanceadas. Con SMOTE-NC, el Recall se eleva a 0.4741 y el F1 a 0.1818, confirmando que el balanceo sintético es una condición necesaria para que el modelo utilice la información de la clase minoritaria en lugar de colapsar hacia la clase mayoritaria.

Sin embargo, se observa un matiz relevante que enriquece la discusión: tanto el AUC-ROC (0.6176 → 0.5940) como el PR-AUC (0.1265 → 0.1145) **disminuyen levemente** con el balanceo. Esto es consistente con la naturaleza de estas dos métricas: ambas evalúan la capacidad de *ranking* del modelo sobre todos los umbrales posibles, independientemente del umbral de decisión que finalmente se use. SMOTE-NC no mejora esa capacidad de ordenamiento subyacente — de hecho, al generar vecinos sintéticos, introduce cierto ruido adicional en las fronteras de decisión — pero sí resuelve el problema práctico de que, con el umbral por defecto (0.5), un modelo entrenado sobre datos desbalanceados nunca cruza ese umbral para la clase minoritaria. Este matiz refuerza la recomendación de Saito y Rehmsmeier (2015), quienes argumentan que la curva Precision-Recall (y su área, PR-AUC) es más informativa que la curva ROC en datasets con fuerte desbalance de clases, precisamente porque no se ve favorecida por la abundancia de verdaderos negativos: en este caso, ambas métricas coinciden en señalar que la capacidad discriminativa *global* del modelo no mejoró, aunque su utilidad *práctica* (medida por Recall y F1) sí lo hizo sustancialmente.

**HE2** ("SMOTE-NC incrementa significativamente el Recall y el F1-Score frente al entrenamiento con clases desbalanceadas") se considera **confirmada**, con la salvedad metodológica de que el efecto de SMOTE-NC debe entenderse como una corrección del punto de operación del modelo (qué tan dispuesto está a predecir la clase minoritaria bajo el umbral estándar), no como una mejora en su capacidad discriminativa intrínseca.

---

## Fase 3: Benchmark de modelos (5 algoritmos, validación cruzada 5-fold)

Se evaluaron los 5 algoritmos mediante validación cruzada estratificada de 5 particiones (*folds*), aplicando SMOTE-NC de forma independiente dentro de cada partición de entrenamiento para evitar fuga de información entre folds — una fila nunca participa simultáneamente como dato real de entrenamiento y como base para generar vecinos sintéticos de otra partición.

| Modelo | AUC-ROC | F1 (clase 1) | F1 Macro | F1 Ponderado | Recall (clase 1) |
|---|---|---|---|---|---|
| SVM | 0.592 ± 0.010 | **0.193 ± 0.006** | 0.473 | 0.702 | 0.508 ± 0.027 |
| Random Forest | 0.601 ± 0.009 | 0.192 ± 0.007 | 0.476 | 0.708 | 0.495 ± 0.038 |
| Gradient Boosting | **0.601 ± 0.010** | 0.187 ± 0.011 | 0.482 | 0.724 | 0.450 ± 0.028 |
| KNN | 0.569 ± 0.010 | 0.181 ± 0.008 | 0.477 | 0.721 | 0.438 ± 0.017 |
| XGBoost | 0.579 ± 0.007 | 0.164 ± 0.009 | **0.509** | **0.793** | 0.271 ± 0.019 |

La matriz de confusión acumulada (suma de los 5 folds, cubriendo la totalidad de las 29,450 observaciones exactamente una vez) confirma el patrón: XGBoost concentra la mayor cantidad de verdaderos negativos (21,453) pero también el menor número de verdaderos positivos (713) y el mayor número de falsos negativos (1,923) — es decir, es el modelo más conservador en detectar pérdida gestacional, precisamente el error que la sensibilidad clínica prioriza evitar.

Este resultado obliga a matizar la hipótesis de superioridad de los ensambles. **HE3** ("los algoritmos basados en ensambles superan de manera estadísticamente significativa en AUROC y F1-Score a los modelos tradicionales") **no se confirma de forma consistente**: por F1 de la clase 1 y por Recall — las métricas clínicamente prioritarias según la literatura citada en la sección 3.4.4 —, el modelo tradicional SVM obtiene el mejor desempeño del benchmark, superando a los 3 ensambles. Por AUC-ROC, ensambles (Random Forest, Gradient Boosting) y SVM quedan prácticamente empatados (diferencia de 0.009, dentro del margen de la desviación estándar entre folds). Solo bajo F1 Macro y F1 Ponderado —métricas que dan mayor peso relativo a la clase mayoritaria por su alta frecuencia— XGBoost aparece como ganador, pero ese resultado es engañoso en el contexto clínico del estudio: su Recall de apenas 0.271 significa que el modelo omite (falso negativo) a más de 7 de cada 10 mujeres con pérdida gestacional real, el error de mayor costo clínico según la propia justificación de la sección de métricas de la tesis. Este hallazgo debe documentarse con honestidad metodológica: la elección de "modelo óptimo" depende críticamente de qué métrica se prioriza, y una selección basada en F1 Ponderado o Accuracy llevaría a una conclusión clínicamente inadecuada.

---

## Fase 4: Explicabilidad global (TreeSHAP)

Se aplicó TreeSHAP (Lundberg et al., 2020) sobre el modelo Random Forest entrenado con el conjunto balanceado, evaluando los valores SHAP sobre la totalidad del conjunto de prueba (8,835 casos, nunca vistos durante el entrenamiento).

| Posición | Variable | Dimensión | % de contribución (mean\|SHAP\|) |
|---|---|---|---|
| 1 | V025 (área de residencia) | Sociodemográfico | 22.1% |
| 2 | V201 (paridad) | Salud | 14.7% |
| 3 | V501 (estado civil) | Sociodemográfico | 13.9% |
| 4 | V106 (nivel educativo) | Sociodemográfico | 11.1% |
| 5 | V222 (intervalo intergenésico) | Salud | 10.8% |
| 6 | V040 (altitud) | Sociodemográfico | 6.6% |
| 7 | IMC | Salud | 5.8% |
| 8 | M14 (controles prenatales) | Salud | 5.8% |
| 9 | V012 (edad) | Sociodemográfico | 5.2% |
| 10 | V190 (índice de riqueza) | Sociodemográfico | 3.9% |

Agregando por dimensión, los **factores sociodemográficos concentran el 62.8%** de la contribución total al modelo, frente a 37.2% de los factores de salud — un resultado que, en principio, respondería directamente a la pregunta de investigación sobre el peso relativo de cada dimensión (PE4).

Es indispensable, sin embargo, contextualizar este resultado frente al de la Fase 1: el ranking de importancia SHAP **difiere sustancialmente** del ranking de Boruta (que ubicaba a V040, V222 e IMC en los primeros lugares, y a V025/V106/V501 entre las variables rechazadas). Esta discrepancia no es un error, sino un hallazgo metodológico esperado y documentado en la literatura: la importancia nativa de Random Forest utilizada por Boruta (Mean Decrease in Impurity) está sujeta a sesgos conocidos, particularmente hacia variables con mayor cardinalidad de categorías o distinta escala de medición (Strobl et al., 2007), mientras que SHAP se basa en valores de Shapley de teoría de juegos, una descomposición aditiva y exacta de la predicción que no comparte ese sesgo (Lundberg et al., 2020). Adicionalmente, el modelo Random Forest explicado en esta fase fue entrenado sobre el conjunto **balanceado** por SMOTE-NC, mientras que el Random Forest usado dentro de Boruta se entrena con `class_weight='balanced'` sobre el conjunto **desbalanceado** original — son, en sentido estricto, dos modelos distintos respondiendo a la misma pregunta, lo que explica parte de la divergencia. Esta comparación entre ambos métodos de importancia debe presentarse en la tesis como una fortaleza metodológica (triangulación de evidencia desde dos ángulos distintos), no como una contradicción a resolver forzosamente a favor de uno u otro.

**HE4** ("los valores TreeSHAP permiten jerarquizar con rigor matemático las variables... según su contribución promedio") se considera **confirmada**: TreeSHAP produjo una jerarquización completa, matemáticamente fundamentada y reproducible, con la dimensión sociodemográfica como determinante dominante del riesgo estimado por el modelo óptimo.

---

## Fase 5: Explicabilidad local (LIME y SHAP Waterfall)

Se analizaron 3 perfiles del conjunto de prueba mediante LIME (Ribeiro et al., 2016) y SHAP Waterfall (Lundberg y Lee, 2017), seleccionados por representar situaciones clínicamente distintas: un caso de alto riesgo correctamente identificado (probabilidad predicha = 0.7675, verdadero positivo), un caso de bajo riesgo correctamente identificado (probabilidad = 0.0145, verdadero negativo) y un caso límite (probabilidad = 0.5000, el punto de mayor incertidumbre del modelo).

En el caso de **alto riesgo**, ambos métodos coinciden en identificar el área de residencia (V025) y el estado civil (V501) como los principales factores que elevan la probabilidad estimada, seguidos por la paridad (V201) y el intervalo intergenésico (V222) — consistente con la jerarquía global de la Fase 4. En el caso de **bajo riesgo**, el estado civil (V501) domina la explicación en dirección opuesta, con una contribución SHAP de -0.3945, la magnitud individual más grande observada en los tres casos analizados. En el **caso límite**, la paridad (V201) es la variable que más empuja la predicción hacia abajo (-0.1698 en SHAP; LIME coincide con -0.1189 para el mismo predicado "V201 > 3"), mientras que el resto de variables empujan levemente hacia arriba, resultando en el equilibrio exacto observado (probabilidad = 0.5000).

La convergencia entre LIME (aproximación lineal local sobre vecinos sintéticos) y SHAP Waterfall (descomposición aditiva exacta) en los tres casos —coincidiendo en las variables de mayor peso, aunque con magnitudes numéricas no directamente comparables por partir de fundamentos matemáticos distintos— constituye evidencia de robustez de la explicación: dos métodos independientes, con supuestos y mecanismos de cálculo diferentes, señalan a las mismas variables como determinantes del riesgo individual estimado.

**HE5** ("las técnicas LIME y SHAP Waterfall permiten descomponer y justificar con precisión las predicciones individuales... para la interpretación de casos clínicos específicos") se considera **confirmada**: ambas técnicas produjeron explicaciones interpretables, coherentes entre sí y trazables hasta el valor base del modelo, cumpliendo el estándar de explicabilidad local que la matriz de consistencia exige para uso clínico.

---

## Síntesis general

De las 5 hipótesis específicas de la matriz de consistencia: **3 se confirman** (HE2, HE4, HE5), **1 se confirma parcialmente** (HE1), y **1 no se confirma de forma consistente** (HE3, superioridad de ensambles), con hallazgos honestos y documentados en cada caso que aportan valor metodológico independientemente del sentido de la confirmación. El hallazgo más relevante para la discusión general de la tesis es la discrepancia entre la jerarquía de variables de Boruta (Fase 1, dominada por factores de salud/ambientales) y la de SHAP (Fase 4, dominada por factores sociodemográficos) — una discrepancia explicable y documentada en la literatura de explicabilidad de modelos de árboles (Strobl et al., 2007; Lundberg et al., 2020), que en sí misma constituye un argumento metodológico a favor de complementar la selección de variables basada en importancia nativa con técnicas de explicabilidad post-hoc como TreeSHAP.
