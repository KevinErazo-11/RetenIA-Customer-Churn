"""Ejecuta TODO el trabajo experimental, sin intervención manual en la selección.

Uso: python run_experiments.py
Prueba mecánica (NO apta como resultado): python run_experiments.py --quick
No se generan resultados inventados si no se puede conseguir el CSV real.
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
import time
import platform
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, precision_recall_curve
from src.core import load_telco,split_data,CSV_NAME,PROJECT_ROOT,SEED
from src.experiments import (score_frame,lr_pipe,rf_pipe,optimize_single,optimize_nsga,
                              choose_model,test_metrics)
from src.xai import create_xai_artifacts
from src.business import simulate_retention

OUT=PROJECT_ROOT/'artifacts'; FIG=PROJECT_ROOT/'figures'


def safe_json(obj):
    if isinstance(obj,dict): return {str(k):safe_json(v) for k,v in obj.items()}
    if isinstance(obj,(tuple,list)): return [safe_json(v) for v in obj]
    if isinstance(obj,np.ndarray): return safe_json(obj.tolist())
    if isinstance(obj,(np.integer,)): return int(obj)
    if isinstance(obj,(np.floating,)): return float(obj)
    if isinstance(obj,(np.bool_,)): return bool(obj)
    if obj is None: return None
    if isinstance(obj,float) and not np.isfinite(obj): return None
    return obj


def dump(obj,name):
    (OUT/name).write_text(json.dumps(safe_json(obj),indent=2,ensure_ascii=False),encoding='utf-8')


def plot_comparison(base, search, history, multi, pareto, pick):
    candidates=pd.concat([base.assign(estrategia=base['modelo']),
                          search[['estrategia','f1_cv','roc_auc_cv']]],ignore_index=True)
    fig,ax=plt.subplots(figsize=(10,5.5))
    ax.barh(candidates['estrategia'],candidates['f1_cv'],color='#147f87')
    ax.set(xlabel='F1 medio en CV estratificada (clase Churn=1)',title='Modelos base y optimizaciones (TRAIN/CV)')
    ax.set_xlim(0,1);fig.tight_layout();fig.savefig(FIG/'comparacion_f1_cv.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    ax.bar(search['estrategia'],search['tiempo_busqueda_s'],color='#244866')
    ax.set(ylabel='Tiempo de optimización (segundos)',title='Coste computacional por estrategia')
    fig.tight_layout();fig.savefig(FIG/'tiempos_optimizacion.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    for method,d in history.groupby('estrategia'):
        ordered=d.sort_values('evaluacion').copy()
        if 'es_evaluacion_completa' in ordered.columns:
            mask=ordered['es_evaluacion_completa'].astype(str).str.lower().eq('true')
            ordered=ordered[mask]
        if not ordered.empty:
            ax.plot(ordered['evaluacion'],ordered['f1_cv'].cummax(),marker='.',label=method)
    ax.set(xlabel='Configuraciones propuestas',ylabel='Mejor F1-CV completo acumulado',title='Convergencia de búsquedas monoobjetivo')
    ax.legend();fig.tight_layout();fig.savefig(FIG/'convergencia_busqueda.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5.5))
    dominated=multi[~multi.no_dominada]
    ax.scatter(dominated['latencia_ms_cliente'],dominated['f1_cv'],c='#9ca3af',alpha=.7,label='Dominadas')
    ax.scatter(pareto['latencia_ms_cliente'],pareto['f1_cv'],c='#007f85',s=70,label='No dominadas (Pareto)')
    sel=multi[multi.trial==pick['trial']].iloc[0]
    ax.scatter([sel.latencia_ms_cliente],[sel.f1_cv],marker='*',s=250,c='#e99c2b',edgecolor='black',label='Elegida')
    ax.set(xlabel='Latencia de inferencia (ms por cliente, menor mejor)',
           ylabel='F1 validación cruzada (mayor mejor)',title='NSGA-II | Espacio de objetivos')
    ax.legend();fig.tight_layout();fig.savefig(FIG/'frente_pareto.png',dpi=170);plt.close(fig)


def run(csv=None,n_trials=16,quick=False):
    OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
    started=time.perf_counter()
    df,audit=load_telco(csv)
    if quick:
        # Solo útil en pruebas locales del programa, NUNCA como resultado de la tarea.
        df=df.groupby('Churn',group_keys=False).sample(frac=.25,random_state=SEED).sort_index()
        print('ADVERTENCIA: QUICK es para pruebas de funcionamiento; NO representa métricas finales.')
    if not quick and len(df)!=7043:
        raise ValueError(f'El dataset oficial debe tener 7043 filas, se encontraron {len(df)}.')
    csv_out=PROJECT_ROOT/'data'/CSV_NAME
    csv_out.parent.mkdir(parents=True,exist_ok=True)
    if not csv_out.exists():shutil.copy2(audit['csv'],csv_out)
    X_train,X_test,y_train,y_test=split_data(df)
    audit.update({'train_rows':len(X_train),'test_rows':len(X_test),'random_state':SEED,
                  'cv_folds':3,'train_churn_rate':float(y_train.mean())})
    dump(audit,'eda_auditoria.json')
    (OUT/'variable_schema.json').write_text(json.dumps({
        'features':list(X_train.columns),'dtypes':{k:str(X_train[k].dtype) for k in X_train.columns},
        'target':'Churn=1','threshold':.5},indent=2),encoding='utf-8')
    print('EDA verificado:',audit['rows'],'clientes | Train',len(X_train),'Test reservado',len(X_test))
    base=pd.DataFrame([
        score_frame('Logistic Regression',lr_pipe(),X_train,y_train),
        score_frame('Random Forest (base)',rf_pipe(),X_train,y_train),
    ]).rename(columns={k:k+'_cv' for k in ['accuracy','precision','recall','f1','roc_auc']})
    base.to_csv(OUT/'comparacion_baselines_cv.csv',index=False)
    print('Modelos base (CV):\n',base.to_string(index=False))
    search,hist=optimize_single(X_train,y_train,n_trials=n_trials)
    search_export=search.copy()
    search_export['params']=search_export['params'].apply(lambda d:json.dumps(d,sort_keys=True))
    search_export.to_csv(OUT/'comparacion_optimizadores_cv.csv',index=False)
    hist.to_csv(OUT/'historial_busquedas.csv',index=False)
    print('Tres estrategias con validación cruzada (Hyperband permite poda):\n',search_export[['estrategia','f1_cv','tiempo_busqueda_s','evaluaciones']].to_string(index=False))
    multi,pareto,pick=optimize_nsga(X_train,y_train,n_trials=n_trials)
    multi.to_csv(OUT/'nsga_todos.csv',index=False)
    pareto.to_csv(OUT/'frente_pareto.csv',index=False)
    dump(pick,'seleccion_pareto.json')
    selection=choose_model(base,search,pick)
    dump(selection,'seleccion_modelo_final.json')
    plot_comparison(base,search,hist,multi,pareto,pick)
    print('Selección congelada ANTES de test:',selection['nombre'],'F1-CV',selection['f1_cv'])
    # No hay cambios ni reajuste en función del rendimiento del conjunto de prueba.
    final=rf_pipe(selection['params']).fit(X_train,y_train)
    logistic=lr_pipe().fit(X_train,y_train)
    base_rf=rf_pipe().fit(X_train,y_train)
    joblib.dump(final,OUT/'modelo_final_pipeline.joblib',compress=3)
    joblib.dump(logistic,OUT/'logistic_regression_pipeline.joblib',compress=3)
    joblib.dump(base_rf,OUT/'random_forest_base_pipeline.joblib',compress=3)
    joblib.dump(X_train.sample(n=min(450,len(X_train)),random_state=SEED),OUT/'xai_background_train.joblib',compress=3)
    # Evaluación final separada (solo tras congelar la selección).
    evaluations={
        'modelo_final': test_metrics(final,X_test,y_test),
        'logistic_regression_baseline':test_metrics(logistic,X_test,y_test),
        'random_forest_baseline':test_metrics(base_rf,X_test,y_test),
    }
    dump(evaluations,'metricas_test_final.json')
    proba=final.predict_proba(X_test)[:,1]
    ConfusionMatrixDisplay.from_predictions(y_test,proba>=.5,labels=[0,1],
                                             display_labels=['Permanece','Abandona'],cmap='Blues')
    plt.title('Matriz de confusión | test aislado | modelo final')
    plt.tight_layout();plt.savefig(FIG/'matriz_confusion_test.png',dpi=170);plt.close()
    fpr,tpr,_=roc_curve(y_test,proba)
    fig,ax=plt.subplots(figsize=(7,5))
    ax.plot(fpr,tpr,label=f"AUC={evaluations['modelo_final']['roc_auc']:.3f}")
    ax.plot([0,1],[0,1],'--',color='#9ca3af');ax.legend()
    ax.set(xlabel='Tasa falsos positivos',ylabel='Recall',title='ROC | test aislado')
    fig.tight_layout();fig.savefig(FIG/'roc_test.png',dpi=170);plt.close(fig)
    precision,recall,_=precision_recall_curve(y_test,proba)
    fig,ax=plt.subplots(figsize=(7,5))
    ax.plot(recall,precision)
    ax.set(xlabel='Recall',ylabel='Precision',title='Precision-Recall | test aislado')
    fig.tight_layout();fig.savefig(FIG/'pr_test.png',dpi=170);plt.close(fig)
    print('RESULTADOS FINALES TEST (sin volver a seleccionar):\n',json.dumps(evaluations['modelo_final'],indent=2))
    xai=create_xai_artifacts(final,X_train,X_test,OUT)
    print('SHAP, LIME y feature importance generados para dos clientes, sin reentrenamiento.')
    # Historial por cliente sobre el dataset demo completo, usado SOLO después del test final.
    all_features=df[X_train.columns].copy()
    prob_full=final.predict_proba(all_features)[:,1]
    clients=df[['customerID','Contract','InternetService','PaymentMethod','tenure','MonthlyCharges','Churn']].copy()
    clients['probabilidad_churn']=prob_full
    clients.to_csv(OUT/'riesgos_clientes_demo.csv',index=False)
    selected,financial=simulate_retention(clients)
    dump(financial,'escenario_comercial_hipotetico.json')
    selected.to_csv(OUT/'priorizacion_demo.csv',index=False)
    manifest={
        'estudiante':'Kevin Andrés Erazo Yepez', 'dataset':'blastchar/telco-customer-churn',
        'dataset_sha256':audit['sha256'],'fecha_ejecucion_utc':datetime.now(timezone.utc).isoformat(),
        'semilla':SEED,'version_python':sys.version.split()[0],
        'modelo_final':selection,'n_trials':n_trials,'ejecucion_quick_no_entregable':bool(quick),
        'optimizadores':['Grid Search','Random Search','Hyperband (Optuna)','NSGA-II'],
        'limitaciones': ['Dataset histórico sintético/de demostración; no validar generalización temporal sin datos de producción.',
                         'Latencia depende del hardware; no interpretar su segundo decimal como propiedad universal.',
                         'SHAP y LIME no prueban causalidad; eficacia de campañas es una hipótesis comercial.']}
    dump(manifest,'manifiesto_experimento.json')
    print('PROYECTO GENERADO | segundos totales',round(time.perf_counter()-started,1))
    return {'auditoria':audit,'baselines_cv':base,'optimizadores_cv':search,'historial':hist,
            'nsga':multi,'pareto':pareto,'seleccion_pareto':pick,'seleccion_final':selection,
            'test_final':evaluations,'xai':xai,'comercial':financial}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--csv',type=str,default=None,help='Ruta del CSV original si está disponible')
    parser.add_argument('--quick',action='store_true',help='Prueba de código, NO es resultado final')
    parser.add_argument('--trials',type=int,default=16,help='16: presupuesto por defecto')
    args=parser.parse_args()
    if args.trials<4:parser.error('--trials debe ser >=4')
    run(args.csv,n_trials=args.trials,quick=args.quick)
