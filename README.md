# FaroCX

**De la probabilidad de abandono a una prioridad de contacto justificable.**

FaroCX es una plataforma de inteligencia comercial para equipos de retención en telecomunicaciones. Combina análisis de cartera, estimación de riesgo individual, explicaciones SHAP y LIME y simulación de campañas sujetas a un presupuesto. Su diferencia está en convertir una alerta predictiva en un **escenario de decisión**: a quién evaluar primero, qué señales explican el riesgo y cuánto costaría actuar.

> **Propuesta central:** no todas las alertas merecen el mismo esfuerzo comercial. FaroCX permite analizar el beneficio neto esperado por contacto bajo supuestos editables, sin presentar esos supuestos como resultados observados.

## Una plataforma orientada a la acción

| Capacidad | Decisión que facilita |
|---|---|
| Radar de cartera | Identificar segmentos con distinta incidencia histórica de churn. |
| Riesgo individual | Estimar la probabilidad de abandono para un perfil concreto. |
| Explicaciones SHAP y LIME | Comprender las señales que influyeron en la predicción. |
| Planificador de campañas | Priorizar candidatos mediante costo por contacto, margen, eficacia supuesta y presupuesto. |
| Transparencia del modelo | Revisar métricas, matriz de confusión, alternativas y frente de Pareto. |

El planificador utiliza un escenario simplificado de beneficio esperado por cliente, basado en probabilidad de abandono, eficacia incremental **hipotética**, margen y horizonte de análisis, menos costo de contacto. Este cálculo **no es una medición causal de clientes retenidos**.

## Resultados de referencia

Evaluación experimental con el conjunto público *Telco Customer Churn*: **7.043 registros**, **19 variables predictoras**, partición estratificada **80/20** y validación cruzada de tres particiones en entrenamiento.

| Clasificador | F1 en test | ROC-AUC en test |
|---|---:|---:|
| Logistic Regression (benchmark) | 0,604 | 0,842 |
| Random Forest seleccionado mediante NSGA-II | 0,581 | 0,830 |

Logistic Regression obtuvo mejor F1 y ROC-AUC en el conjunto de prueba. El Random Forest se seleccionó **antes** de abrir el test, mediante una regla de compromiso entre F1 en validación y latencia de inferencia. Esta selección no demuestra que sea la mejor alternativa para producción; ambas deberían compararse nuevamente con datos de la operadora.

Para optimizar Random Forest se emplearon **Grid Search, Random Search y Hyperband (Optuna)**; **NSGA-II** exploró simultáneamente F1 y latencia. Random Search y Hyperband alcanzaron F1-CV de 0,5736 en esta ejecución, con menor tiempo para Random Search.

## Arquitectura

`CSV de telecomunicaciones → EDA → ColumnTransformer / Pipeline → Logistic Regression y Random Forest → HPO / NSGA-II → modelos exportados con joblib → SHAP / LIME → Streamlit`

El portal no vuelve a entrenar el modelo al atender una consulta. El preprocesamiento, la clasificación y los artefactos de análisis quedan separados de la capa web.

## Ejecutar localmente

Requiere Python compatible, dependencias de `requirements.txt` y los artefactos de modelo de `artifacts/` incluidos en el repositorio.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

La presentación comercial de cinco diapositivas se genera con `python build_pitch.py` y LaTeX Beamer (`pdflatex`), usando los resultados guardados. No es necesario reentrenar para generar el PDF.

## Adopción responsable

FaroCX es una demostración funcional sobre datos históricos públicos. Las predicciones son estimaciones estadísticas; SHAP, LIME y Feature Importance no demuestran causalidad. Los resultados económicos de campañas son escenarios basados en supuestos, no reducciones observadas de churn. Antes de operar con datos privados se requieren controles de acceso, revisión de privacidad, validación temporal y un piloto A/B con métricas acordadas.

**FaroCX — Menos contactos a ciegas. Más decisiones justificables.**
