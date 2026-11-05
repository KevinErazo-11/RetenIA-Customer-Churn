"""Notebook reproducible de Tarea 7, conservando exactamente las 33 celdas del EDA.

La cadena experimental se ejecuta UNA vez desde run_experiments.run(), mientras
las celdas siguientes leen artefactos creados por esa ejecución. Así Streamlit,
Beamer y notebook usan el mismo resultado y no hay selección post-test.
"""
import copy
from pathlib import Path
import nbformat as nb

ROOT = Path(__file__).resolve().parent
EDA = ROOT / 'notebooks' / '01_EDA_Telco_Customer_Churn.ipynb'
DEST = ROOT / 'notebooks' / '02_Tarea7_Proyecto_Completo_Kevin_Erazo.ipynb'
eda = nb.read(EDA, as_version=4)
new = nb.v4.new_notebook()
M = lambda source: nb.v4.new_markdown_cell(source)
C = lambda source: nb.v4.new_code_cell(source)
new.cells.append(M('''# Tarea 7 · Optimización de Machine Learning, Explainable AI y Business Pitch

**Kevin Andrés Erazo Yepez — Aprendizaje Automático**  
**Dataset:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).  
**Producto:** FaroCX · inteligencia comercial de retención.

Este notebook reproduce los requisitos en el orden del enunciado: preparación,
modelos, Grid Search, Random Search, Hyperband, NSGA-II, SHAP, LIME,
Streamlit y propuesta de negocio. Se conservan sin alterar las **33 celdas del
EDA original** de este mismo proyecto.

**Protocolo:** split 80/20 estratificado (seed 42), folds solo dentro de TRAIN,
selección por F1-CV y latencia predefinida, y TEST reservado para evaluación final.
Los resultados del conjunto de prueba **no** cambian la configuración elegida.
El umbral de clasificación es 0.50 y la clase positiva es `Churn=1`.
'''))
new.cells.append(C('''# Resolver el directorio de trabajo tanto desde la raíz como desde notebooks/.
import os, sys
from pathlib import Path
PROJECT = next((q for p in [Path.cwd(), *Path.cwd().parents] for q in [p]
                if (q / 'src' / 'core.py').is_file()), None)
if PROJECT is None:
    raise FileNotFoundError('Abre la carpeta V2 completa en VS Code: deben existir src/ y data/.')
os.chdir(PROJECT)
if str(PROJECT) not in sys.path: sys.path.insert(0, str(PROJECT))
print('Raíz del proyecto:', PROJECT)
'''))
new.cells.append(M('''# Parte 1 · Exploratory Data Analysis (EDA)

El siguiente bloque corresponde al EDA original de esta conversación. Incluye
identificación de variables, auditoría de `TotalCharges`, balance de `Churn`,
exploración de entrenamiento y la especificación de `ColumnTransformer` y `Pipeline`.
El EDA no ajusta el preprocesador sobre el holdout.
'''))
new.cells.extend(copy.deepcopy(eda.cells))
new.cells.append(M('''---
# Parte 1 · Clasificadores y evaluación reproducible

Se comparan dos familias: **Logistic Regression** (baseline interpretable) y
**Random Forest** (relaciones no lineales). Accuracy es informativa pero
insuficiente con una prevalencia de abandono minoritaria. Se reportan
Precision, Recall, F1, ROC-AUC y matriz de confusión de `Churn=1`.

La siguiente celda realiza **una única ejecución integral**: divide los datos,
compara clasificadores por CV, busca hiperparámetros, calcula NSGA-II, congela
la selección, evalúa TEST y exporta los modelos y las explicaciones. El resto
del notebook lee esas mismas evidencias, sin volver a entrenar.
'''))
new.cells.append(C('''from run_experiments import run
from src.core import CSV_NAME, FEATURES, NUMERIC, CATEGORICAL
from pathlib import Path
import json, pandas as pd, numpy as np
from IPython.display import display, Markdown, Image

csv_file = PROJECT / 'data' / CSV_NAME
report = run(csv=str(csv_file) if csv_file.exists() else None, n_trials=16, quick=False)
ART = PROJECT / 'artifacts'
FIG = PROJECT / 'figures'
read_json=lambda name: json.loads((ART/name).read_text(encoding='utf-8'))
print('La cadena experimental exportó sus resultados a:', ART)
'''))
new.cells.append(C('''audit = read_json('eda_auditoria.json')
print('Registros:', audit['rows'], ' | Churn=1:', audit['churn_yes'],
      ' | Churn=0:', audit['churn_no'])
print('Train:', audit['train_rows'], ' | Test aislado:', audit['test_rows'])
print('Variables numéricas:', NUMERIC)
print('Variables categóricas:', CATEGORICAL)
display(pd.DataFrame([audit]).T.rename(columns={0:'Auditoría'}))
'''))
new.cells.append(C('''baseline = pd.read_csv(ART/'comparacion_baselines_cv.csv')
display(baseline.round(4))
print('Interpretación: comparamos los clasificadores con F1-CV y ROC-AUC-CV sin utilizar TEST.')
'''))
new.cells.append(M(r'''## Métricas y matriz de confusión

Para la clase positiva `Churn=1`:

$$\mathrm{Precision}=\frac{TP}{TP+FP}, \qquad
\mathrm{Recall}=\frac{TP}{TP+FN}, \qquad
F1=2\frac{\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}.$$

- **FN:** cliente que abandona pero no fue identificado; oportunidad de retención perdida.
- **FP:** cliente que permanece pero recibe una alerta; posible gasto comercial innecesario.
- **ROC-AUC:** discriminación del ranking a través de diferentes umbrales.

El conjunto TEST se consulta **solo después** de congelar la selección; estos resultados no
son usados para modificar hiperparámetros ni umbral.
'''))
new.cells.append(C('''metrics=read_json('metricas_test_final.json')
rows=[]
for name,m in metrics.items():
    rows.append({'Modelo':name,**{key:m[key] for key in ('accuracy','precision','recall','f1','roc_auc','average_precision')}})
display(pd.DataFrame(rows).set_index('Modelo').round(4))
print('Matriz de confusión del modelo final (TN,FP / FN,TP):')
print(np.array(metrics['modelo_final']['matriz_confusion']))
for name in ('matriz_confusion_test.png','roc_test.png','pr_test.png'):
    display(Image(filename=str(FIG/name)))
'''))
new.cells.append(M('''# Parte 2 · Hyperparameter Optimization

Se optimiza **el mismo Random Forest**, con idéntico esquema `ColumnTransformer`
+ `Pipeline`, tres folds estratificados fijos y F1 como objetivo.

- **Grid Search:** recorre exhaustivamente la cuadrícula de 16 combinaciones.
- **Random Search:** evalúa 16 configuraciones aleatorias de un espacio más amplio.
- **Hyperband (Optuna):** combina múltiples asignaciones de recursos y parada
  anticipada (*early stopping*). Cada configuración aumenta de **20 → 60 → 180 árboles**,
  con `warm_start`, permitiendo suspender aquellas que no justifican más cómputo.
  Optuna implementa Hyperband con un `HyperbandPruner` y `RandomSampler`.

**Comparabilidad:** los tres métodos proponen 16 configuraciones y usan los mismos folds,
pero Grid Search utiliza una cuadrícula más estrecha. Hyperband usa presupuestos
*incrementales*, de modo que el número de configuraciones plenamente evaluadas,
los árboles construidos y el tiempo total **no son idénticos**. Se reportan ambos
presupuestos para evitar comparaciones engañosas.
'''))
new.cells.append(C('''from src.experiments import GRID, GLOBAL
from sklearn.model_selection import ParameterGrid
space = pd.DataFrame({
    'Hiperparámetro': [key.removeprefix('modelo__') for key in GRID],
    'Grid Search': [str(values) for values in GRID.values()],
    'Random Search / NSGA-II': [str(GLOBAL[key]) for key in GRID],
    'Hyperband': [
        'Recurso: 20, 60, 180 árboles (máximo 180)',
        str(GLOBAL['modelo__max_depth']),
        str(GLOBAL['modelo__min_samples_leaf']),
        str(GLOBAL['modelo__max_features'])]
})
display(space)
print('Grid Search:', len(ParameterGrid(GRID)), 'combinaciones.')
print('CV: StratifiedKFold(n_splits=3, shuffle=True, random_state=42).')
'''))
new.cells.append(C('''search=pd.read_csv(ART/'comparacion_optimizadores_cv.csv')
cols=['estrategia','f1_cv','recall_cv','roc_auc_cv','tiempo_busqueda_s',
      'evaluaciones','configuraciones_completas','configuraciones_podadas',
      'recurso_total_arboles_por_fold','equivalentes_180_arboles']
display(search[cols].round(4))
print('La mejor estrategia por F1 de CV es:',
      search.loc[search['f1_cv'].idxmax(), 'estrategia'])
'''))
new.cells.append(C('''history=pd.read_csv(ART/'historial_busquedas.csv')
display(history[[c for c in ('estrategia','evaluacion','f1_cv',
    'estado','recurso_arboles','es_evaluacion_completa') if c in history.columns]].head(30).round(4))
for name in ('comparacion_f1_cv.png','tiempos_optimizacion.png','convergencia_busqueda.png'):
    display(Image(filename=str(FIG/name)))
'''))
new.cells.append(M(r'''# Parte 3 · Multiobjective Optimization: NSGA-II

**Maximizamos la calidad** y **minimizamos el tiempo de inferencia** del pipeline
completo sobre los mismos folds de TRAIN:

$$\max_{\theta}f_1(\theta)=\frac{1}{3}\sum_{k=1}^{3}F1_k(\theta),$$

$$\min_{\theta}f_2(\theta)=\operatorname{mediana}_k\left[
\frac{1000 T_{\mathrm{pred},k}(\theta)}{n_{\mathrm{clientes},k}}
\right]\quad[\mathrm{ms/cliente}].$$

La latencia tiene calentamiento previo, cinco repeticiones por fold y hasta 128 clientes
por lote. No corresponde al tiempo de entrenamiento y depende del hardware.

Un punto es **no dominado** si no hay otra solución que mejore o empate ambos
objetivos y mejore estrictamente al menos uno. La regla previa al test acepta
soluciones a ≤0,01 puntos del mejor F1 Pareto para escoger menor latencia, y
prefiere dicha solución si está a ≤0,015 puntos del mejor Random Forest monoobjetivo.
'''))
new.cells.append(C('''all_nsga=pd.read_csv(ART/'nsga_todos.csv')
pareto=pd.read_csv(ART/'frente_pareto.csv')
choice=read_json('seleccion_pareto.json')
selection=read_json('seleccion_modelo_final.json')
print('Opciones NSGA-II:',len(all_nsga),' | Frente Pareto:',len(pareto))
display(pareto[['trial','f1_cv','latencia_ms_cliente','roc_auc_cv']].round(5))
print('Solución Pareto propuesta:', choice)
print('MODELO OPERATIVO SELECCIONADO POR CV:',selection)
display(Image(filename=str(FIG/'frente_pareto.png')))
'''))
new.cells.append(M('''# Parte 4 · Explainable AI: SHAP, LIME y Feature Importance

**SHAP global** mide la contribución media absoluta de características originales
(agregación de One-Hot). **SHAP local** suma contribuciones respecto de una
probabilidad base para explicar `P(Churn=1)`.

**LIME local** perturba las variables originales, incluida la codificación de
categorías, y aproxima el comportamiento alrededor del cliente. Se explica a
**los mismos dos perfiles** para las dos técnicas: mayor y menor riesgo estimado
en el holdout. La elección no utiliza la etiqueta real del cliente.

**Random Forest Feature Importance** utiliza reducción media de impureza; puede
favorecer determinadas características y no proporciona dirección causal.
Ninguna de estas técnicas demuestra que cambiar un atributo evite bajas.
'''))
new.cells.append(C('''shap=pd.read_csv(ART/'shap_global.csv')
fi=pd.read_csv(ART/'rf_feature_importance.csv')
display(shap.head(10).round(4))
display(fi.head(10).round(4))
for name in ('shap_global.png','rf_feature_importance.png'):
    display(Image(filename=str(ART/name)))
'''))
new.cells.append(C('''examples=read_json('explicaciones_clientes.json')
display(pd.DataFrame(examples)[['cliente','indice_original','probabilidad','base_shap','shap_residual']])
for i in (1,2):
    display(Markdown(f'### Cliente {i}: SHAP y LIME'))
    display(pd.read_csv(ART/f'shap_cliente_{i}.csv').head(8))
    display(pd.read_csv(ART/f'lime_cliente_{i}.csv').head(8))
    display(Image(filename=str(ART/f'shap_cliente_{i}.png')))
    display(Image(filename=str(ART/f'lime_cliente_{i}.png')))
'''))
new.cells.append(M('''# Parte 5 · Streamlit

`app.py` es el producto comercial FaroCX. Integra **Exploratory Data Analysis**
(con visualizaciones, KPIs y filtros), **Model Performance**, **Hyperparameter
Optimization**, **NSGA-II · Pareto Frontier**, **Explainable AI · SHAP y LIME**,
**Prediction & Explainability** y **Business Application**.

La aplicación **no entrena**; carga pipelines `.joblib`, métricas y gráficos exportados.
El público no ve instrucciones de clase, un guion de exposición ni controles internos.
Para ejecutarla desde el directorio raíz: `python -m streamlit run app.py`.
'''))
new.cells.append(M(r'''# Parte 6 · Propuesta comercial FaroCX

**Problema:** una operadora tiene recursos limitados para contactar a clientes,
pero el abandono provoca pérdida potencial de margen.

**Producto y comprador:** FaroCX es una plataforma de riesgo, explicación y
priorización de campañas para gerencias de CRM/retención.

**Diferenciadores:** auditoría reproducible, Grid Search, Random Search,
Hyperband, NSGA-II, explicaciones SHAP/LIME y simulación de presupuesto.

**Monetización propuesta:** licencia hipotética de **USD 59/mes** más servicios
de integración, cuyos costos reales aún deben cotizarse.

**Valor neto esperado bajo hipótesis** por cliente:

$$V_i=p_i\,e\,M_i\,m\,H-c_i,$$

donde $p_i$ es el riesgo predicho, $e$ eficacia incremental **no medida**,
$M_i$ mensualidad, $m$ margen hipotético, $H$ meses y $c_i$ costo de contacto.
La suma del escenario considera también la licencia.

**Riesgos:** sesgo, privacidad, deriva temporal, datos no representativos,
latencia variable y coste de falsos negativos/falsos positivos. Una prueba
**A/B** mediría el efecto incremental real de las campañas antes de escalar.
'''))
new.cells.append(C('''economics=read_json('escenario_comercial_hipotetico.json')
display(pd.DataFrame([economics]).T.rename(columns={0:'Valor del escenario (hipotético)'}))
print('No corresponde a ROI causal estimado; se basa en supuestos editables.')
'''))
new.cells.append(M('''# Entrega y Business Pitch

- Notebook reproducible con celdas de EDA original y evidencias experimentales.
- Modelos exportados, código y `requirements.txt` para repositorio GitHub.
- Streamlit comercial funcionando y enlace público comprobado antes de Moodle.
- Beamer/LaTeX: **máximo cinco diapositivas, fondo claro y sin código**.
- Negociación presencial: 2 minutos de propuesta/demostración, 2 minutos de
  preguntas y 1 minuto de cierre. El turno como comprador también se evalúa.

**Observación científica:** el resultado independiente puede favorecer a
Logistic Regression incluso si el seleccionado por CV/NSGA-II es Random Forest.
No debe reemplazarse el modelo después de ver TEST: se informa la diferencia.
'''))
new.cells.append(C('''# Crear PDF Beamer desde los resultados de esta ejecución; requiere pdflatex instalado.
from build_pitch import main as generate_pitch
generate_pitch()
print('Revisar las cinco diapositivas y ejecutar python VALIDAR_ENTREGA.py.')
'''))
new.cells.append(C('''# Conclusiones generadas con las MÉTRICAS REALES de la nueva ejecución.
final=metrics['modelo_final']
logit=metrics['logistic_regression_baseline']
best=search.sort_values('f1_cv',ascending=False).iloc[0]
print(f"Modelo seleccionado antes del holdout: {selection['nombre']} (F1-CV={selection['f1_cv']:.4f}).")
print(f"Test final: Accuracy={final['accuracy']:.4f}; Precision={final['precision']:.4f}; "
      f"Recall={final['recall']:.4f}; F1={final['f1']:.4f}; ROC-AUC={final['roc_auc']:.4f}.")
print(f"Mejor optimización monoobjetivo: {best['estrategia']} (F1-CV={best['f1_cv']:.4f}).")
print(f"Logistic Regression test: F1={logit['f1']:.4f}; compararlo honestamente con el modelo final.")
print('SHAP top-3:', ', '.join(shap['Variable'].head(3)))
print('Límites: los factores no son causas y el ROI comercial es un escenario hipotético.')
'''))
new.metadata = {
    'kernelspec': {'display_name':'Python 3 (FaroCX)', 'language':'python','name':'python3'},
    'language_info': {'name':'python'},
}
DEST.parent.mkdir(exist_ok=True,parents=True)
nb.write(new, DEST)
print('Notebook construido:',DEST,'| celdas:',len(new.cells),'| EDA original:',len(eda.cells))
