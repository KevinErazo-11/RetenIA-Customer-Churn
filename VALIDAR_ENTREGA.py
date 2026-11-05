"""Valida que existen artefactos REALES, sin pruebas quick y sin inventar resultados."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).parent
expected=[
 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv',
 'artifacts/modelo_final_pipeline.joblib','artifacts/logistic_regression_pipeline.joblib',
 'artifacts/random_forest_base_pipeline.joblib','artifacts/metricas_test_final.json',
 'artifacts/manifiesto_experimento.json','artifacts/frente_pareto.csv',
 'artifacts/shap_global.png','artifacts/shap_cliente_1.png','artifacts/shap_cliente_2.png',
 'artifacts/lime_cliente_1.png','artifacts/lime_cliente_2.png',
 'figures/matriz_confusion_test.png','figures/frente_pareto.png',
 'notebooks/02_Tarea7_Proyecto_Completo_Kevin_Erazo.ipynb','app.py',
 'Pitch_FaroCX_Kevin_Erazo.pdf', 'Pitch_FaroCX_Kevin_Erazo.tex']
missing=[x for x in expected if not (ROOT/x).exists()]
if missing:
 print('NO COMPLETA: faltan estos artefactos reales:')
 for x in missing: print(' -',x)
 sys.exit(1)
manifest=json.loads((ROOT/'artifacts/manifiesto_experimento.json').read_text())
if manifest.get('ejecucion_quick_no_entregable'):
 print('NO COMPLETA: modelos quick (no aptos para calificación)');sys.exit(2)
# La presentación oficial es Beamer/PDF, con máximo 5 páginas.
try:
    import fitz
    doc = fitz.open(ROOT/'Pitch_FaroCX_Kevin_Erazo.pdf')
    if len(doc) != 5:
        print('NO COMPLETA: el PDF debe tener exactamente cinco diapositivas.');sys.exit(3)
    print('Presentación PDF LaTeX: cinco diapositivas verificadas.')
except ImportError:
    print('PDF presente; verificar manualmente cinco páginas (PyMuPDF no instalado).')
print('Estructura local validada. AÚN falta verificar manualmente GitHub y la URL pública de Streamlit.')
