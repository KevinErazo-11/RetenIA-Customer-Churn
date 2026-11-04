# RetenIA | Proyecto final de Aprendizaje Automático

**Autor:** Kevin Andrés Erazo Yepez. **Dataset:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

Versión reconstruida desde cero con **Grid Search, Random Search, Hyperband (Optuna), NSGA-II, SHAP, LIME**, una aplicación Streamlit comercial y un **Business Pitch Beamer/LaTeX en PDF**. Conserva el EDA original (33 celdas) y las decisiones de limpieza. No incluye resultados de entrenamientos anteriores ni métricas inventadas.

## Inicio limpio en Ubuntu

La raíz del proyecto es la carpeta **V2/**. Antes de reemplazar tu V2 anterior, **haz una copia de seguridad** para conservar tus resultados previos. Descomprime el ZIP final en una carpeta temporal y mueve la nueva `V2` a la ubicación deseada.

```bash
source "/home/kevin-erazo/Documentos/Ambiente Virtual/bin/activate"
PROJECT="/home/kevin-erazo/OneDrive/8) Octavo Semestre/Aprendizaje Automatico/7. Optimización de Machine Learning, Explainable AI y Business Pitch/V2"
cd "$PROJECT"
python -m pip install -r requirements.txt
python -m pip check
python run_experiments.py
python build_pitch.py
python VALIDAR_ENTREGA.py
python -m streamlit run app.py
```

Si Kaggle no descarga automáticamente los datos, copia **`WA_Fn-UseC_-Telco-Customer-Churn.csv`** a `data/` desde el enlace oficial y vuelve a ejecutar `python run_experiments.py`. Para desarrollar el notebook: ejecuta `python build_notebook.py` (ya se incluye el `.ipynb` construido), selecciona tu kernel y usa **Run All**; esta acción vuelve a entrenar todo. Para evitar entrenar dos veces, basta con el script para servir la app; ejecuta el notebook una vez cuando necesites guardar las salidas de la entrega.

**No uses `--quick` en la entrega.** Su único propósito es una prueba de software.

## Recorrido comercial por Streamlit

1. Inicio — síntesis de cartera y valor de RetenIA.
2. Exploratory Data Analysis — KPIs, segmentación y visualizaciones interactivas.
3. Model Performance — Logistic Regression, Random Forest, modelo seleccionado, métricas y matriz de confusión.
4. Hyperparameter Optimization — **Grid Search, Random Search, Hyperband (Optuna)**. Se ven CV, F1, tiempos, recursos, candidatos y podas.
5. NSGA-II · Pareto Frontier — F1-CV frente a latencia y compromiso operativo.
6. Explainable AI · SHAP y LIME — visión global, dos perfiles y Random Forest Feature Importance.
7. Prediction & Explainability — formulario para cliente individual, predicción y explicaciones.
8. Business Application — costos, prioridades y escenarios comerciales hipotéticos.
9. Información del servicio — límites, privacidad y uso responsable.

Las pantallas se expresan en tono **comercial y técnico para el cliente**, sin guiones escolares ni menciones al profesor. Modelos se leen desde `.joblib`; **Streamlit no reentrena**.

## Metodología y honestidad experimental

- 7.043 registros del CSV original; limpieza de `TotalCharges`, exclusión de `customerID`, 3 características numéricas y 16 categóricas; `ColumnTransformer` y `Pipeline`.
- `train_test_split` estratificado 80/20 y `StratifiedKFold(3)` solo en train; test aislado hasta cerrar la selección.
- **Grid Search** exhaustivo: 16 configuraciones en una cuadrícula reducida.
- **Random Search**: 16 configuraciones del espacio global.
- **Hyperband (Optuna)**: 16 candidatos nominales, recursos 20 → 60 → 180 árboles, `HyperbandPruner`, `RandomSampler` y `warm_start` dentro de cada fold. Puede podar ensayos; se reportan candidatos completos, podados y árboles consumidos. **No** se afirma presupuesto computacional idéntico.
- **NSGA-II**: F1-CV máximo y tiempo de inferencia mínimo; soluciones no dominadas y regla de selección publicada antes de test.
- Comparación independiente contra **Logistic Regression** y **Random Forest** en test: si el baseline supera al seleccionado, se declara y no se cambia el modelo tras mirar test.
- **SHAP/LIME** para `Churn=1`, dos clientes del holdout; explicación no implica causalidad.
- Las cifras económicas son **hipótesis**, no ahorro ni ROI probado; un piloto A/B sería necesario para medir el impacto causal.

## Resultados y archivos

`artifacts/` contiene métricas CSV/JSON, pipelines `.joblib` y explicaciones; `figures/` las gráficas; `data/` el CSV original; `notebooks/02_Tarea7_Proyecto_Completo_Kevin_Erazo.ipynb` la evidencia reproducible. `build_pitch.py` crea `Pitch_RetenIA_Kevin_Erazo.tex` y `Pitch_RetenIA_Kevin_Erazo.pdf` **después de entrenar**, con las métricas de la nueva ejecución. Se requieren `pdflatex` y un PDF de **exactamente 5 páginas**, fondo claro, sin código.

El despliegue remoto en Streamlit Community Cloud y el repositorio GitHub se deben publicar y comprobar con las credenciales de su propietario; este ZIP no puede contener una URL pública que no existe.

## Documentación de la defensa

La guía técnica y el libreto para la negociación se mantienen en `docs/`, **fuera de la interfaz comercial**.
