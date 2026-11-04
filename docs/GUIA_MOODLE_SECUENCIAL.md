# Guía técnica privada para presentar RetenIA

Este archivo es solo para estudiar y preparar la defensa, **no** se muestra en Streamlit.

## Parte 1 · EDA y modelos (1 punto)
Explicar dataset de 7.043 filas y 19 predictores; la clase positiva es `Churn=1`. El cliente no debe confundir precisión con accuracy. Preprocesamiento dentro de `Pipeline` y partición 80/20 estratificada. Comparar Logistic Regression y Random Forest con Accuracy, Precision, Recall, F1, ROC-AUC y matriz. Para un falso negativo, se pierde la oportunidad de retener; falso positivo, contacto innecesario.

## Parte 2 · Hyperparameter Optimization (1,5)
**Grid Search** (rejilla de 16), **Random Search** (16 muestras) y **Hyperband (Optuna)** (16 candidatos nominales, recursos 20/60/180 árboles, poda). Las tres usan mismos folds; **no** mismo espacio ni necesariamente el mismo cómputo. Explicar tabla de hiperparámetros, media de F1-CV, tiempo de búsqueda, candidatos completados y podados.

## Parte 3 · NSGA-II (1)
Objetivos: maximizar F1 en CV y minimizar ms/cliente. Pareto = no dominado. No confundir tiempo de entrenamiento con tiempo de inferencia; recordar latencia dependiente del hardware. Explicar la selección F1 ≤0,01 del óptimo para elegir latencia, tolerancia 0,015 para adoptar frente a mejor RF monoobjetivo.

## Parte 4 · SHAP y LIME (1,5)
SHAP global y local sobre los mismos dos perfiles que LIME. Clase positiva `Churn=1`. Comparar importancia global de SHAP con Feature Importance por impureza de Random Forest. Interpretar dirección local, no causalidad.

## Parte 5 · Streamlit (2)
Mostrar producto RetenIA al comprador: EDA con filtros/gráficos/KPIs; comparación y Pareto; perfil de cliente con probabilidad y explicación; escenario de campañas. El portal NO muestra este guion ni menciona a Moodle o al profesor, y NO ajusta modelos.

## Parte 6 · Propuesta de negocio (1,5)
Cliente potencial: equipo de CRM de una operadora. Valor: priorizar contactos. Licencia hipotética USD 59/mes. Diferenciar riesgo predictivo de ahorro causal. Proponer piloto A/B para medir bajas evitadas y retorno neto.

## Business Pitch (1,5)
Cinco diapositivas en PDF Beamer fondo claro. Duración: 2 minutos presentación + demo, 2 minutos preguntas del comprador, 1 minuto respuestas/cierre. Como vendedor frente a Elvis Ismael Yangari Calva; como comprador frente a Ismael Adrián Paredes Rosero. La calidad de preguntas como comprador también cuenta.

## Checklist
- [ ] Ejecutar `python run_experiments.py` (sin `--quick`).
- [ ] Revisar métricas reales, no inventar ni copiar de una ejecución anterior.
- [ ] Ejecutar y guardar el notebook completo.
- [ ] `python build_pitch.py` y `pdfinfo ... | grep Pages` debe marcar 5.
- [ ] Ejecutar `python VALIDAR_ENTREGA.py`.
- [ ] Verificar las nueve pantallas de Streamlit.
- [ ] Publicar GitHub con modelos/artefactos y URL accesible a Streamlit.
