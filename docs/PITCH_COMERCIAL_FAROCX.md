# FaroCX | Guion comercial y manejo de objeciones

## Mensaje de venta

**No vendemos una predicción aislada. Vendemos un método para decidir a qué clientes dedicar recursos comerciales con información y supuestos explícitos.**

**Comprador:** gerencia de CRM y retención de una operadora.

**Diferencia concreta:** el riesgo se conecta con un criterio de priorización económica simulado: `probabilidad_de_abandono × eficacia_hipotética × margen × horizonte − costo_de_contacto`. SHAP y LIME permiten interpretar la alerta. La eficacia no es causal ni está demostrada; se propone un piloto controlado.

## Venta de 2 minutos (con demo)

**0:00–0:20 — Problema.** «Una operadora no puede destinar el mismo presupuesto a todos. En la cartera histórica de referencia, el 26,5 % terminó en baja. ¿Cómo elegimos a quién contactar primero?»

**0:20–0:45 — Solución.** «Presento FaroCX. Estima el riesgo individual, explica las señales del modelo y permite ordenar una campaña por beneficio neto esperado bajo un presupuesto. El punto diferencial no es solo detectar bajas: es pasar de una alerta a una decisión comercial que pueda revisarse.»

**0:45–1:20 — Demo.** «En Prediction & Explainability introduzco un perfil. Aquí aparece una probabilidad y su explicación SHAP/LIME. Después voy a Business Application: puedo cambiar el costo por contacto y el presupuesto, observar cómo cambia la cartera priorizada y exportarla. Estos son escenarios, no bajas evitadas.»

**1:20–1:40 — Evidencia.** «Nuestro Random Forest seleccionado previamente obtuvo ROC-AUC de 0,830 sobre test reservado. Comparamos modelos: Logistic Regression alcanzó 0,842, y lo mostramos abiertamente. Para un piloto real mediríamos las alternativas con sus propios datos.»

**1:40–2:00 — Propuesta.** «Propongo un piloto acotado, con métricas de calidad, costo de contacto y resultados de retención medidos contra un grupo de control. El precio preliminar ilustrativo es USD 59/mes, ajustable. ¿Podemos acordar alcance, presupuesto y fecha de revisión?»

## Objeciones: respuestas breves

**«¿Me garantiza reducir churn?»** «No sería serio garantizarlo. El modelo predice riesgo; el impacto de las intervenciones se valida con un A/B suficientemente largo.»

**«¿Qué ofrece frente a otro dashboard?»** «Además de clasificar, FaroCX conecta riesgos con costos, márgenes, presupuesto y explicaciones locales. El resultado es una lista priorizada para revisar, no solo un gráfico.»

**«¿Por qué su Logistic Regression es mejor que el modelo NSGA-II?»** «Es cierto en test: F1 0,604 frente a 0,581. La selección de Random Forest se cerró antes de test con un criterio de F1 y latencia. El piloto compararía ambos y seleccionaría según los costos reales del negocio.»

**«¿59 dólares incluye implementación y garantía?»** «No. Es una hipótesis preliminar de licencia, no una cotización contractual. El alcance técnico, la integración y la seguridad deben presupuestarse por separado.»

**«¿Y la privacidad?»** «La demostración usa datos públicos. Una instalación empresarial necesitaría permisos, autenticación, protección de datos y revisión de cumplimiento antes de conectar una cartera privada.»

## Cierre de 1 minuto

«No le propongo comprar una promesa de ingresos. Le propongo un ensayo medible con reglas claras: una cartera autorizada, un presupuesto delimitado, responsables de CRM y métricas compartidas. Si los resultados de la intervención justifican el costo, negociamos el despliegue. Si no, no escalamos. ¿Podemos dejar agendada la definición de ese piloto?»

## Demostración segura

Abrir previamente la app pública. En `Prediction & Explainability` mostrar un perfil de prueba; luego en `Business Application` variar costo y presupuesto. Evitar cargar datos privados. Mantener preparado el PDF ante una incidencia de conectividad.
