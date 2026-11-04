import numpy as np
import pandas as pd
from src.core import NUMERIC, CATEGORICAL
from src.experiments import optimize_single


def test_grid_random_hyperband_con_poda_y_presupuesto():
    rng=np.random.default_rng(6)
    n=90
    frame=pd.DataFrame({column:rng.choice(['Yes','No'],n) for column in CATEGORICAL})
    for col in NUMERIC:
        frame[col]=rng.uniform(1,100,n)
    label=pd.Series(np.r_[np.zeros(n//2,dtype=int),np.ones(n-n//2,dtype=int)])
    scores, history=optimize_single(frame,label,n_trials=4)
    assert set(scores['estrategia']) == {'Grid Search','Random Search','Hyperband (Optuna)'}
    assert all(scores['evaluaciones'] == 4)
    assert (scores['configuraciones_completas'] >= 1).all()
    assert (scores['configuraciones_completas'] + scores['configuraciones_podadas'] == scores['evaluaciones']).all()
    assert 'es_evaluacion_completa' in history
    assert 'recurso_arboles' in history
    assert (scores['recurso_total_arboles_por_fold'] > 0).all()
