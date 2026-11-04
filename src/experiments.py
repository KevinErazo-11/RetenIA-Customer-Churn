"""Comparación experimental. Todas las búsquedas usan pipeline y mismos folds.

La optimización/selección jamás usa X_test/y_test, salvo en evaluate_test_once()
cuando el modelo final ya está elegido. Los F1 se calculan para clase 1.
"""
from __future__ import annotations
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, average_precision_score)
from sklearn.model_selection import ParameterGrid
from .core import SEED, make_preprocessor, cv_splitter

optuna.logging.set_verbosity(optuna.logging.WARNING)

SCORING = {'accuracy':'accuracy', 'precision':'precision', 'recall':'recall',
           'f1':'f1', 'roc_auc':'roc_auc'}
BASE_RF = {'n_estimators':100, 'max_depth':None,'min_samples_leaf':1,
           'max_features':'sqrt'}
# Cuadrícula exhaustiva, limitada a 16 configuraciones para presupuesto equiparable.
GRID = {
    'modelo__n_estimators': [60,180],
    'modelo__max_depth': [6,14],
    'modelo__min_samples_leaf': [1,4],
    'modelo__max_features': ['sqrt',.75]
}
# Espacio global: 7 * 7 * 5 * 3 = 735 configuraciones posibles.
GLOBAL = {
    'modelo__n_estimators': [40,60,80,100,120,160,180],
    'modelo__max_depth': [4,6,8,10,14,20,None],
    'modelo__min_samples_leaf': [1,2,3,4,6],
    'modelo__max_features': ['sqrt',.5,.75]
}


def rf_pipe(params: dict | None = None):
    model_params = dict(BASE_RF)
    if params:
        model_params.update({k.removeprefix('modelo__'):v for k,v in params.items()})
    model = RandomForestClassifier(**model_params, random_state=SEED, n_jobs=1)
    return Pipeline([('prep',make_preprocessor()),('modelo',model)])


def lr_pipe():
    return Pipeline([('prep', make_preprocessor()), ('modelo',
                     LogisticRegression(max_iter=2000, class_weight=None, random_state=SEED))])


def cv_metrics(pipeline, X, y):
    scores = cross_validate(pipeline, X, y, cv=cv_splitter(), scoring=SCORING,
                            return_train_score=False, n_jobs=1, error_score='raise')
    return {metric:float(np.mean(scores['test_'+metric])) for metric in SCORING}


def score_frame(name, pipeline, X, y):
    t0=time.perf_counter()
    results=cv_metrics(pipeline,X,y)
    results.update({'modelo':name,'tiempo_cv_s':round(time.perf_counter()-t0,3)})
    return results


def _collect_sklearn(name, search, X, y):
    t0=time.perf_counter()
    search.fit(X,y)
    elapsed=time.perf_counter()-t0
    table=pd.DataFrame(search.cv_results_)
    rows=[]
    for idx,row in table.iterrows():
        entry={'estrategia':name,'evaluacion':int(idx+1),'f1_cv':float(row['mean_test_f1']),
               'f1_std':float(row['std_test_f1']), 'accuracy_cv':float(row['mean_test_accuracy']),
               'precision_cv':float(row['mean_test_precision']), 'recall_cv':float(row['mean_test_recall']),
               'roc_auc_cv':float(row['mean_test_roc_auc']),
               'tiempo_fit_cv_s':float(row['mean_fit_time'])*3,
               'estado':'COMPLETE','es_evaluacion_completa':True,
               'recurso_arboles':int(row['params']['modelo__n_estimators'])}
        entry['params']=json.dumps(row['params'],sort_keys=True)
        rows.append(entry)
    winner_idx=int(table['mean_test_f1'].idxmax())
    winner=table.iloc[winner_idx]
    summary={'estrategia':name, 'f1_cv':float(winner['mean_test_f1']),
             'f1_std':float(winner['std_test_f1']),
             'accuracy_cv':float(winner['mean_test_accuracy']),
             'precision_cv':float(winner['mean_test_precision']),
             'recall_cv':float(winner['mean_test_recall']),
             'roc_auc_cv':float(winner['mean_test_roc_auc']),
             'tiempo_busqueda_s':float(elapsed),'evaluaciones':int(len(table)),
             'folds':3,'params':winner['params'],
             'configuraciones_completas':int(len(table)),
             'configuraciones_podadas':0,
             'recurso_total_arboles_por_fold':int(sum(row['modelo__n_estimators'] for row in table['params'])),
             'equivalentes_180_arboles':float(sum(row['modelo__n_estimators'] for row in table['params']) / 180.0)}
    return summary,rows


def _hyperband_cv_objective(trial, X, y):
    """Hyperband real con recursos = árboles, tres folds fijos y warm_start.

    Se ajusta el preprocesamiento únicamente en el TRAIN de cada fold. Cada
    bosque crece de 20 -> 60 -> 180 árboles sobre los mismos datos/fold, por
    lo que el presupuesto medido corresponde a árboles efectivamente construidos.
    """
    params = {
        'max_depth': trial.suggest_categorical('max_depth', GLOBAL['modelo__max_depth']),
        'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', GLOBAL['modelo__min_samples_leaf']),
        'max_features': trial.suggest_categorical('max_features', GLOBAL['modelo__max_features']),
    }
    prepared = []
    for i_train, i_val in cv_splitter().split(X, y):
        xa, xb = X.iloc[i_train], X.iloc[i_val]
        ya, yb = y.iloc[i_train], y.iloc[i_val]
        prep = make_preprocessor().fit(xa, ya)
        model = RandomForestClassifier(n_estimators=20, max_depth=params['max_depth'],
                     min_samples_leaf=params['min_samples_leaf'],
                     max_features=params['max_features'], random_state=SEED,
                     n_jobs=1, warm_start=True)
        prepared.append((prep.transform(xa), prep.transform(xb), ya.to_numpy(), yb.to_numpy(), model))
    last_metrics = None
    for trees in (20, 60, 180):
        fold = {k: [] for k in SCORING}
        for xa, xb, ya, yb, model in prepared:
            model.set_params(n_estimators=trees)
            model.fit(xa, ya)
            proba = model.predict_proba(xb)[:, 1]
            predicted = (proba >= .5).astype(int)
            fold['accuracy'].append(accuracy_score(yb, predicted))
            fold['precision'].append(precision_score(yb, predicted, zero_division=0))
            fold['recall'].append(recall_score(yb, predicted, zero_division=0))
            fold['f1'].append(f1_score(yb, predicted, zero_division=0))
            fold['roc_auc'].append(roc_auc_score(yb, proba))
        last_metrics = {name: float(np.mean(arr)) for name, arr in fold.items()}
        last_metrics['f1_std'] = float(np.std(fold['f1']))
        trial.set_user_attr('ultimo_recurso_arboles', trees)
        trial.set_user_attr('metricas_ultimo_recurso', last_metrics)
        trial.report(last_metrics['f1'], step=trees)
        if trial.should_prune():
            raise optuna.TrialPruned(f'Hyperband: parada anticipada a {trees} árboles')
    return last_metrics['f1']


def optimize_single(X, y, n_trials=16):
    """Grid Search, Random Search y Hyperband (Optuna). No utiliza test.

    Presupuesto comparable por número de candidatos propuestos. Hyperband
    emplea recursos incrementales y puede podar ensayos: el consumo efectivo
    se reporta por separado, sin afirmar igualdad computacional exacta.
    """
    if n_trials != 16:
        grid_candidates = list(ParameterGrid(GRID))[:n_trials]
        param_grid = [{k: [v] for k, v in entry.items()} for entry in grid_candidates]
    else:
        param_grid = GRID
    grid = GridSearchCV(rf_pipe(), param_grid=param_grid, scoring=SCORING, refit=False,
                        cv=cv_splitter(), n_jobs=1, return_train_score=False, error_score='raise')
    random = RandomizedSearchCV(rf_pipe(), param_distributions=GLOBAL,
              n_iter=n_trials, random_state=SEED, scoring=SCORING, refit=False,
              cv=cv_splitter(), n_jobs=1, return_train_score=False, error_score='raise')
    summary_grid, hist_grid = _collect_sklearn('Grid Search', grid, X, y)
    summary_random, hist_random = _collect_sklearn('Random Search', random, X, y)
    study = optuna.create_study(direction='maximize',
                sampler=optuna.samplers.RandomSampler(seed=SEED),
                pruner=optuna.pruners.HyperbandPruner(
                    min_resource=20, max_resource=180, reduction_factor=3))
    t0 = time.perf_counter()
    study.optimize(lambda trial: _hyperband_cv_objective(trial, X, y),
                   n_trials=n_trials, n_jobs=1, show_progress_bar=False)
    elapsed = time.perf_counter() - t0
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
                and t.user_attrs.get('ultimo_recurso_arboles') == 180]
    if not complete:
        raise RuntimeError('Hyperband no completó configuraciones con recurso máximo; aumenta --trials')
    best = max(complete, key=lambda t: t.value)
    metrics = best.user_attrs['metricas_ultimo_recurso']
    final_params = {'modelo__' + k: v for k, v in best.params.items()}
    final_params['modelo__n_estimators'] = 180
    resources = [int(t.user_attrs.get('ultimo_recurso_arboles', 0)) for t in study.trials]
    summary_hb = {
        'estrategia': 'Hyperband (Optuna)',
        'f1_cv': float(best.value),
        'f1_std': metrics['f1_std'],
        **{k + '_cv': v for k, v in metrics.items() if k not in ('f1', 'f1_std')},
        'tiempo_busqueda_s': float(elapsed), 'evaluaciones': len(study.trials),
        'configuraciones_completas': len(complete),
        'configuraciones_podadas': len(study.trials) - len(complete),
        'recurso_total_arboles_por_fold': sum(resources),
        'equivalentes_180_arboles': sum(resources) / 180.,
        'folds': 3, 'params': final_params,
    }
    hist_hb = []
    for t in study.trials:
        values = t.user_attrs.get('metricas_ultimo_recurso', {})
        hist_hb.append({
            'estrategia': 'Hyperband (Optuna)', 'evaluacion': int(t.number + 1),
            'f1_cv': values.get('f1', float('nan')),
            'f1_std': values.get('f1_std', float('nan')),
            **{k + '_cv': values.get(k, float('nan')) for k in SCORING if k != 'f1'},
            'estado': t.state.name,
            'es_evaluacion_completa': t.state == optuna.trial.TrialState.COMPLETE,
            'recurso_arboles': t.user_attrs.get('ultimo_recurso_arboles', 0),
            'params': json.dumps({'modelo__' + k: v for k, v in t.params.items()}, sort_keys=True),
        })
    return pd.DataFrame([summary_grid, summary_random, summary_hb]), pd.DataFrame(hist_grid + hist_random + hist_hb)


def trial_params(trial):
    # Las mismas cotas y los mismos valores disponibles para Random y NSGA-II.
    return {
        'modelo__n_estimators': trial.suggest_categorical('n_estimators', GLOBAL['modelo__n_estimators']),
        'modelo__max_depth': trial.suggest_categorical('max_depth', GLOBAL['modelo__max_depth']),
        'modelo__min_samples_leaf': trial.suggest_categorical('min_samples_leaf', GLOBAL['modelo__min_samples_leaf']),
        'modelo__max_features': trial.suggest_categorical('max_features', GLOBAL['modelo__max_features'])
    }


def latency_ms_per_customer(pipeline, Xval, batches=5):
    """Latencia por cliente (ms), pipeline completo, mediana de lotes.
    La medición se realiza con lotes fijos y calentamiento. CPU y contención afectan el resultado.
    """
    subset=Xval.iloc[:min(128,len(Xval))]
    pipeline.predict_proba(subset)
    dt=[]
    for _ in range(batches):
        t=time.perf_counter()
        pipeline.predict_proba(subset)
        dt.append((time.perf_counter()-t)*1000/len(subset))
    return float(np.median(dt))


def cv_multiobjectives(pipeline, X, y):
    f1s=[]; runtimes=[]; aucs=[]
    for train_i,val_i in cv_splitter().split(X,y):
        xa=X.iloc[train_i]; xb=X.iloc[val_i]
        ya=y.iloc[train_i]; yb=y.iloc[val_i]
        pipe=clone(pipeline).fit(xa,ya)
        proba=pipe.predict_proba(xb)[:,1]
        f1s.append(f1_score(yb,proba>=.5,zero_division=0))
        aucs.append(roc_auc_score(yb,proba))
        runtimes.append(latency_ms_per_customer(pipe,xb))
    return float(np.mean(f1s)),float(np.median(runtimes)),float(np.mean(aucs))


def optimize_nsga(X,y,n_trials=16):
    population_size=8 if n_trials>=16 else 4
    sampler=optuna.samplers.NSGAIISampler(population_size=population_size,seed=SEED)
    study=optuna.create_study(directions=['maximize','minimize'],sampler=sampler)
    t0=time.perf_counter()
    def objective(trial):
        params=trial_params(trial)
        f1,lat,auc=cv_multiobjectives(rf_pipe(params),X,y)
        trial.set_user_attr('auc',auc)
        trial.set_user_attr('parameter_dict',params)
        return f1,lat
    study.optimize(objective,n_trials=n_trials,n_jobs=1,show_progress_bar=False)
    runtime=float(time.perf_counter()-t0)
    best_ids={t.number for t in study.best_trials}
    rows=[]
    for trial in study.trials:
        if trial.values is None: continue
        rows.append({'trial':trial.number,'f1_cv':float(trial.values[0]),
                     'latencia_ms_cliente':float(trial.values[1]),
                     'roc_auc_cv':float(trial.user_attrs['auc']),
                     'no_dominada':trial.number in best_ids,
                     'params':json.dumps(trial.user_attrs['parameter_dict'],sort_keys=True)})
    df=pd.DataFrame(rows)
    pareto=df[df.no_dominada].sort_values(['f1_cv','latencia_ms_cliente'],ascending=[False,True]).copy()
    # Regla declarada antes de consultar el test: tolerancia F1 <= 0.01, elegir latencia menor.
    max_f1=float(pareto.f1_cv.max())
    eligible=pareto[pareto.f1_cv >= max_f1 - .01]
    pick=eligible.sort_values(['latencia_ms_cliente','f1_cv','trial'],ascending=[True,False,True]).iloc[0]
    chosen={
        'trial':int(pick.trial),'f1_cv':float(pick.f1_cv),'latencia_ms_cliente':float(pick.latencia_ms_cliente),
        'params':json.loads(pick.params),'tiempo_busqueda_s':runtime,
        'evaluaciones':int(len(df)),'poblacion':population_size,
        'criterio':'F1 dentro de 0.01 del máximo de Pareto; escoger menor latencia (desempate F1).'
    }
    return df,pareto,chosen


def choose_model(base_models:pd.DataFrame, search:pd.DataFrame, nsga_choice:dict):
    """Selección solo mediante CV; RF óptimo puede venir de cualquier método.
    Regla: frente Pareto preferido si su F1 está a <=0.015 del mejor RF mono-objetivo;
    si no, mejor RF mono-objetivo (mayor F1, luego menor tiempo de búsqueda).
    LR se mantiene como benchmark, sin cambiar familia durante optimización.
    """
    sorted_search=search.sort_values(['f1_cv','tiempo_busqueda_s'],ascending=[False,True])
    leader=sorted_search.iloc[0]
    if nsga_choice['f1_cv'] >= float(leader.f1_cv)-.015:
        return {'nombre':'NSGA-II (Pareto)','params':nsga_choice['params'],
                'f1_cv':nsga_choice['f1_cv'], 'motivo':'F1 a menos de 0.015 del mejor RF y menor latencia elegible en Pareto.'}
    params=leader.params if isinstance(leader.params,dict) else json.loads(leader.params)
    return {'nombre':str(leader.estrategia),'params':params,'f1_cv':float(leader.f1_cv),
            'motivo':'Mejor F1 en CV; solución Pareto no alcanzó la tolerancia de 0.015.'}


def test_metrics(trained_pipeline,X_test,y_test):
    proba=trained_pipeline.predict_proba(X_test)[:,1]
    pred=(proba>=.5).astype(int)
    return {
        'accuracy':float(accuracy_score(y_test,pred)),
        'precision':float(precision_score(y_test,pred,zero_division=0)),
        'recall':float(recall_score(y_test,pred,zero_division=0)),
        'f1':float(f1_score(y_test,pred,zero_division=0)),
        'roc_auc':float(roc_auc_score(y_test,proba)),
        'average_precision':float(average_precision_score(y_test,proba)),
        'matriz_confusion':confusion_matrix(y_test,pred,labels=[0,1]).tolist(),
        'test_size':int(len(y_test)), 'threshold':.5,
        'evaluacion':'Una evaluación en holdout final; no se ha usado para seleccionar el modelo.'
    }
