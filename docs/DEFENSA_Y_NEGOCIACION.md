# Defensa técnica y comercial — Tarea 7

**Estudiante:** Kevin Andrés Erazo Yepez · Grupo 3, código D.

## Mensaje central

**FaroCX** transforma señales de clientes en prioridades de contacto: estima la probabilidad de abandono, explica por qué el modelo asigna el riesgo y permite simular un presupuesto limitado. Es una herramienta de **apoyo a decisiones**, no un sistema que demuestre qué acción causa retención. Antes de vender un retorno, proponemos probarlo con un piloto controlado.

## Pitch de 2 minutos (guion sugerido)

**0:00–0:22. Problema.** «Una operadora no puede contactar con la misma intensidad a todos sus clientes. El dataset Telco tiene 7.043 registros y una minoría de clientes que abandonan, alrededor del 26,5 %. Un clasificador que predice que nadie abandona tendría una Accuracy alta, pero sería inútil para retención».

**0:22–0:49. Producto.** «Presento FaroCX, un panel con cuatro apartados: exploración de cartera, rendimiento de modelos, predicción explicable de nuevos clientes y priorización comercial. Un agente puede introducir los atributos de un cliente y consultar su probabilidad de abandono y los factores de riesgo del modelo».

**0:49–1:17. Credibilidad.** «No decidimos por Accuracy solamente: comparamos Logistic Regression y Random Forest en validación cruzada y medimos Precision, Recall, F1 y ROC-AUC. Random Forest es la familia optimizada con Grid Search, Random Search, Hyperband y NSGA-II. El conjunto de prueba estuvo reservado hasta cerrar la elección; los resultados específicos deben mostrarse desde el artefacto real generado».

**1:17–1:40. Diferenciación.** «NSGA-II compara F1 frente a tiempo de inferencia. SHAP y LIME explican casos individuales; así un equipo puede justificar una priorización sin afirmar causalidad. El simulador añade presupuesto, coste de campaña y margen estimado».

**1:40–2:00. Oferta y siguiente paso.** «Propongo una licencia demostrativa desde USD 59 mensuales, precio aún hipotético. Antes de recomendar una inversión definitiva, sugiero un piloto A/B para medir bajas evitadas incrementalmente, costes de contacto y cumplimiento de privacidad. ¿Aceptaría evaluar el piloto?»

> **Importante:** incorporar al discurso métricas reales de `artifacts/metricas_test_final.json` y de `artifacts/comparacion_optimizadores_cv.csv` una vez ejecutado el experimento. No citar números de resultados de modelos sin ejecución. El profesor puede preguntar valores concretos.

## Demostración Streamlit durante esos 2 minutos

Mostrar en esta secuencia: (1) EDA y porcentaje churn con filtros; (2) Model Performance con tabla de métricas y Pareto; (3) Prediction & Explainability con un caso predefinido; (4) Business Application ajustando el presupuesto. Para ahorrar tiempo, los widgets deben estar probados previamente y el servidor debe estar caliente.

## Ronda comercial de 5 minutos — asignación

- **Ronda 1, comprador (C→D):** Ismael Adrián Paredes Rosero te presentará su propuesta; pregunta por retorno incremental y prevención de leakage.
- **Ronda 2, vendedor (D→E):** venderás FaroCX a Elvis Ismael Yangari Calva.
- Los estudiantes sin turno deben observar y anotar preguntas; prepara respuestas al docente.

## Preguntas del profesor y respuestas defendibles

### 1. ¿Por qué no usar Accuracy como métrica principal?

Churn está desbalanceado: la clase 1 es minoritaria. Predecir siempre la clase 0 obtiene una Accuracy elevada pero Recall=0 y F1=0 sobre abandono. Por ello optimizamos el F1 de la clase positiva y reportamos Precision, Recall, ROC-AUC y confusión. En una operadora real la decisión óptima también depende de costes de falsos positivos y falsos negativos.

### 2. ¿Por qué Logistic Regression y Random Forest?

Logistic Regression es una referencia interpretable, rápida y sensible al escalado; Random Forest maneja no linealidades e interacciones y tiene hiperparámetros de capacidad y latencia (número de árboles, profundidad, hojas). Se usan para cumplir al menos dos clasificadores y disponer de `Feature Importance` de RF para comparar contra SHAP/LIME.

### 3. ¿Para qué ColumnTransformer y Pipeline?

Unifican imputación por mediana, escalado numérico, imputación por moda y One-Hot para categóricas. El `fit` de cada transformador ocurre **solo dentro de cada fold de entrenamiento** durante la CV. No se calculan medias, medianas ni categorías usando el test.

### 4. ¿Por qué se imputaron once TotalCharges con cero?

En la versión estándar del dataset hay 11 campos de facturación acumulada en blanco para `tenure=0`. El EDA verifica la condición antes de imputar 0, dado que no existe historial de meses facturados. No se eliminan esas filas; los demás valores faltantes inesperados deben investigarse.

### 5. ¿Qué es One-Hot y por qué no label encoding?

Los servicios y métodos de pago son nominales y no tienen orden. Asignar 0,1,2,3 inventaría magnitudes y distancias. One-Hot crea indicadores independientes. `SeniorCitizen` se trata como categoría; `customerID` se excluye para no memorizar identificadores.

### 6. ¿Por qué tres folds y mismo presupuesto?

Tres folds estratificados con semilla 42 mantienen proporciones de Churn y equilibran tiempo computacional. Grid Search y Random Search evalúan 16 configuraciones completas; Hyperband propone 16 candidatos pero puede podar algunos antes del máximo de 180 árboles; NSGA-II explora 16 alternativas multiobjetivo. Grid utiliza una cuadrícula más acotada y las estrategias no consumen necesariamente el mismo cómputo.

### 7. ¿Cuál es la diferencia entre Grid Search, Random Search y Hyperband?

Grid Search enumera todas las combinaciones de la cuadrícula definida; Random Search selecciona configuraciones del espacio global; Hyperband (Optuna) usa recursos crecientes (20, 60 y 180 árboles), poda configuraciones poco prometedoras y reporta tanto candidatos completos como suspendidos. Hyperband usa asignación adaptativa del presupuesto computacional y poda de ensayos; en esta implementación no se realiza optimización bayesiana.

### 8. ¿Qué es NSGA-II y por qué necesitamos Pareto?

NSGA-II es una búsqueda evolutiva multiobjetivo, clasifica por dominancia y fomenta diversidad. Una solución no dominada no puede mejorar F1 sin empeorar latencia con respecto a los candidatos disponibles. Pareto ofrece alternativas que una única métrica escondería. Si los puntos tienen latencias muy cercanas, podemos priorizar estabilidad y F1 sobre diferencias de milésimas de segundo.

### 9. ¿Cómo se mide tiempo de inferencia?

Se mide el `predict_proba` del pipeline completo, con un lote de hasta 128 clientes y calentamiento previo. Se repite cinco veces y se usa mediana por muestra y por fold. **Es dependiente de CPU y no equivale a un SLA de producción**.

### 10. ¿Por qué reservar test?

Si se mirara test para ajustar el modelo, se filtraría información hacia la elección y la evaluación quedaría optimista. Toda selección utiliza CV dentro de train. Test se consulta una sola vez después de congelar parámetros y umbral 0.50; luego se usa únicamente para reportar/explicar.

### 11. ¿Qué significan TN, FP, FN y TP en esta tarea?

Clase positiva=abandono. TP = churn detectado; FN = churn real no detectado (oportunidad perdida); FP = cliente que permanece, marcado como abandono (contacto potencialmente innecesario); TN = permanencia correctamente reconocida.

### 12. ¿Qué indican SHAP y LIME? ¿Son lo mismo?

SHAP atribuye contribuciones respecto a un valor base de la predicción; LIME aproxima localmente el modelo con una regresión ponderada de perturbaciones. SHAP ofrece un método de atribución y resumen global; LIME puede variar por la muestra de perturbaciones. Ambos explican el comportamiento del modelo, no causas reales.

### 13. ¿Por qué la importancia del Random Forest puede diferir de SHAP?

`feature_importances_` agrega reducciones de impureza a través de árboles y puede favorecer variables con más puntos de corte. SHAP mide contribución sobre predicciones para muestras concretas. LIME aproxima regiones locales. Categóricas One-Hot deben reagruparse por variable original para compararlas justamente.

### 14. ¿Cómo presentaría una explicación a un gerente no técnico?

«El sistema estima un riesgo de abandono del **valor que muestre la app**. En el cálculo, el contrato mensual y el historial de servicios pueden aumentar el riesgo, mientras que la permanencia larga podría reducirlo **si el gráfico de este cliente lo confirma**. Estos factores explican la predicción; antes de actuar, un agente debe verificar preferencias y necesidades reales».

### 15. ¿Cómo justificas el precio y el ROI?

USD 59/mes es un **precio hipotético** para discutir la propuesta, no tarifa validada. El simulador muestra `p_abandono × eficacia_supuesta × cargo_mensual × margen_supuesto × horizonte − costo_contacto`, y descuenta la licencia. Sin experimento A/B no sabemos eficacia incremental real. El piloto debe contrastar campañas contra un grupo de control.

### 16. ¿Qué errores pueden dañar a la empresa?

Falsos negativos: perder clientes que abandonan. Falsos positivos: gastar contactos/ofertas en clientes estables y causar fricción. Riesgos adicionales: deriva de datos, sesgos en segmentos, mala calibración, acceso a datos sensibles, falta de consentimiento, procesamiento fuera de contexto y sobrepromesas comerciales.

### 17. ¿Se utilizaron datos simulados para métricas?

Los resultados del trabajo deben provenir **exclusivamente del CSV original**. Las pruebas de desarrollo mecánicas con datos sintéticos no se publican ni se confunden con métricas reales. El modelo final guardado incluye SHA256 de la fuente para trazabilidad.

### 18. ¿Qué harías antes de operar en un telecom real?

Reunir datos recientes, definir horizonte de churn y ventana temporal, controlar disponibilidad de cada variable, comparar con validación temporal, revisar calibración por segmento, probar estrategias en A/B, monitorear drift, medir retorno neto real y auditar privacidad y fairness.

## Respuestas a ocho objeciones comerciales

1. **¿Por qué invertir?** Porque automatiza priorización y muestra el coste de equivocarse; proponemos piloto, no garantía.
2. **¿Qué ventaja?** Pipeline trazable + explicabilidad doble + presupuesto + frente de Pareto.
3. **¿Cómo confiar?** Métricas del holdout y revisión por humanos; nunca 100 % de certeza.
4. **¿Qué si falla?** Matriz de confusión permite cuantificar falsos positivos/negativos; controles de seguimiento.
5. **¿Por qué el modelo?** Mejor compromiso validación F1/latencia según regla ex ante, no elección arbitraria.
6. **¿Cómo demostrar ahorro?** Comparación incremental A/B entre grupo atendido y control.
7. **¿Qué riesgos?** Generalización, privacidad, equidad, deriva, integraciones, calibración y eficacia no identificada.
8. **¿Por qué pagar ese precio?** Se ofrece como hipótesis y sujeto a un piloto con criterios verificables de éxito.

## Minuta para negociar como comprador

Pregunta a Ismael: «¿Qué evidencia del set reservado demuestra generalización sin data leakage?», «¿Qué métrica refleja el coste empresarial de un falso negativo?» y «¿El ROI está basado en un A/B o solo en supuestos?». Evalúa evidencia y originalidad de forma respetuosa.

## Cierre de un minuto

«La decisión no es comprar una promesa de ingresos, sino probar una herramienta medible. FaroCX permite priorizar, explicar y simular costes con transparencia. Propongo un piloto delimitado, con métricas de éxito, control de privacidad y revisión de resultados antes de una suscripción anual. ¿Podemos acordar condiciones para ese piloto?»
