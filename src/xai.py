"""SHAP, LIME y feature importance con mapeo explícito a clase Churn=1.

SHAP TreeExplainer se calcula sobre el Random Forest después de transformar las 19
características. LIME perturba la matriz transformada y solo invoca el clasificador
interno, para evitar volver a aplicar OneHotEncoder a datos ya codificados.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .core import FEATURES,NUMERIC,CATEGORICAL,SEED


def class_1_shap(pipe, X:pd.DataFrame):
    import shap
    matrix=pipe.named_steps['prep'].transform(X)
    names=pipe.named_steps['prep'].get_feature_names_out()
    explainer=shap.TreeExplainer(pipe.named_steps['modelo'])
    shap_result=explainer(matrix,check_additivity=False)
    values=np.asarray(shap_result.values)
    base=np.asarray(shap_result.base_values)
    if values.ndim==3 and values.shape[-1]==2:
        values=values[:,:,1]
        if base.ndim==2: base=base[:,1]
    elif isinstance(shap_result.values,list):
        values=np.asarray(shap_result.values[1]);base=np.repeat(float(explainer.expected_value[1]),len(X))
    elif values.ndim==2:
        if base.ndim==2: base=base[:,1]
    else:
        raise ValueError(f'Forma SHAP inesperada: {values.shape}')
    if base.ndim==0:
        base=np.full(len(X),float(base))
    p=pipe.predict_proba(X)[:,1]
    # TreeExplainer de RF de scikit-learn devuelve contribuciones en escala de probabilidad.
    residual=float(np.max(np.abs(base+values.sum(axis=1)-p)))
    if residual > .03:
        raise ValueError(f'La explicación SHAP no reproduce P(Churn=1): residuo={residual}')
    return {'values':values,'base':base,'matrix':np.asarray(matrix),
            'names':list(map(str,names)), 'probabilities':p, 'residual':residual}


def group_original_features(names, importances):
    mapping=[]
    for name in names:
        if name.startswith('num__'):
            mapping.append(name.removeprefix('num__'))
        elif name.startswith('cat__'):
            remainder=name.removeprefix('cat__')
            matching=[c for c in CATEGORICAL if remainder==c or remainder.startswith(c+'_')]
            mapping.append(max(matching,key=len) if matching else remainder)
        else: mapping.append(name)
    grouped=pd.DataFrame({'origen':mapping,'importance':importances})
    grouped=grouped.groupby('origen',as_index=False)['importance'].sum().sort_values('importance',ascending=False)
    return grouped


def create_xai_artifacts(pipe,X_train,X_test,output_dir:Path,client_ids=None):
    """Produce gráficos y explicaciones de 2 clientes de test, posterior a congelar el modelo."""
    output_dir.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(SEED)
    sampled=X_train.iloc[np.sort(rng.choice(len(X_train),size=min(120,len(X_train)),replace=False))]
    shap_result=class_1_shap(pipe,sampled)
    names=shap_result['names']
    vals=shap_result['values']
    feature_importance=group_original_features(names,np.abs(vals).mean(axis=0))
    feature_importance.columns=['Variable','SHAP_importancia_abs_media']
    feature_importance.to_csv(output_dir/'shap_global.csv',index=False)
    rf=pipe.named_steps['modelo']
    fi=group_original_features(names,rf.feature_importances_)
    fi.columns=['Variable','RandomForest_importancia_impureza']
    fi.to_csv(output_dir/'rf_feature_importance.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,6))
    view=feature_importance.head(12).iloc[::-1]
    ax.barh(view['Variable'],view['SHAP_importancia_abs_media'],color='#007E86')
    ax.set(xlabel='Importancia media absoluta SHAP (probabilidad)',title='Importancia global | SHAP | Churn=1')
    fig.tight_layout();fig.savefig(output_dir/'shap_global.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,6))
    fi_view=fi.head(12).iloc[::-1]
    ax.barh(fi_view['Variable'],fi_view['RandomForest_importancia_impureza'],color='#3C6388')
    ax.set(xlabel='Reducción media de impureza',title='Random Forest | Feature Importance')
    fig.tight_layout();fig.savefig(output_dir/'rf_feature_importance.png',dpi=170);plt.close(fig)
    predictions=pipe.predict_proba(X_test)[:,1]
    # Casos distintos por riesgo extremo, sin atender al verdadero y para seleccionarlos.
    selected_pos=[int(np.argmax(predictions)),int(np.argmin(predictions))]
    selected=X_test.iloc[selected_pos]
    local=class_1_shap(pipe,selected)
    xai_rows=[]
    for i in range(len(selected)):
        effects=local['values'][i]
        rank=np.argsort(np.abs(effects))[::-1][:12]
        frame=pd.DataFrame({'feature':[names[k] for k in rank], 'SHAP_churn_1':[float(effects[k]) for k in rank]})
        frame.to_csv(output_dir/f'shap_cliente_{i+1}.csv',index=False)
        fig,ax=plt.subplots(figsize=(10,5))
        c=['#C45742' if x>0 else '#16857F' for x in frame['SHAP_churn_1'].iloc[::-1]]
        ax.barh(frame['feature'].iloc[::-1],frame['SHAP_churn_1'].iloc[::-1],color=c)
        ax.set(xlabel='Contribución a probabilidad de Churn=1 (+ aumenta riesgo)',title=f'SHAP local | cliente {i+1} | p={predictions[selected_pos[i]]:.3f}')
        ax.axvline(0,color='gray',linewidth=1)
        fig.tight_layout();fig.savefig(output_dir/f'shap_cliente_{i+1}.png',dpi=160);plt.close(fig)
        xai_rows.append({'cliente':i+1,'indice_original':str(selected.index[i]),'probabilidad':float(predictions[selected_pos[i]]),
                         'base_shap':float(local['base'][i]),'shap_residual':float(local['residual'])})
    # LIME en variables ORIGINALES con categóricas discretas. El predictor decodifica
    # categorías y llama a todo el Pipeline. Evita generar combinaciones One-Hot inválidas.
    from lime.lime_tabular import LimeTabularExplainer
    lime_exp, model_predict, transformed_selected = build_lime_explainer(pipe,X_train,selected)
    for i in range(len(selected)):
        exp=lime_exp.explain_instance(transformed_selected[i],model_predict,num_features=12,
                                       labels=[1],num_samples=1200)
        rows=[{'regla_lime':r, 'peso_lime_churn_1':float(v)} for r,v in exp.as_list(label=1)]
        pd.DataFrame(rows).to_csv(output_dir/f'lime_cliente_{i+1}.csv',index=False)
        fig=exp.as_pyplot_figure(label=1)
        fig.set_size_inches(10,5)
        fig.tight_layout();fig.savefig(output_dir/f'lime_cliente_{i+1}.png',dpi=160);plt.close(fig)
        xai_rows[i]['lime_top_positivo']=[r for r in rows if r['peso_lime_churn_1']>0][:3]
        xai_rows[i]['lime_top_negativo']=[r for r in rows if r['peso_lime_churn_1']<0][:3]
    (output_dir/'explicaciones_clientes.json').write_text(json.dumps(xai_rows,indent=2,ensure_ascii=False),encoding='utf-8')
    return xai_rows


def build_lime_explainer(pipe, X_train, X_new):
    """LIME respeta relaciones One-Hot: perturba categóricas originales y usa todo el pipeline."""
    from lime.lime_tabular import LimeTabularExplainer
    # La columna número j se corresponde con FEATURES[j] (NUMERIC primero).
    levels={c:sorted(X_train[c].astype(str).dropna().unique().tolist()) for c in CATEGORICAL}
    def encode(df):
        enc=df[FEATURES].copy()
        for col in NUMERIC: enc[col]=pd.to_numeric(enc[col],errors='raise').astype(float)
        for col in CATEGORICAL:
            pos={name:i for i,name in enumerate(levels[col])}
            mapped=enc[col].astype(str).map(pos)
            if mapped.isna().any():
                raise ValueError(f'Categoría de {col} no observada en entrenamiento')
            enc[col]=mapped.astype(float)
        return enc.to_numpy(dtype=float)
    tr=encode(X_train);pt=encode(X_new)
    def predict_numeric(values):
        arr=np.atleast_2d(np.asarray(values,dtype=float))
        d=pd.DataFrame(arr,columns=FEATURES)
        for col in CATEGORICAL:
            catvals=levels[col]
            ids=np.clip(np.rint(d[col]).astype(int),0,len(catvals)-1)
            d[col]=[catvals[i] for i in ids]
        for col in NUMERIC: d[col]=pd.to_numeric(d[col],errors='raise')
        return pipe.predict_proba(d[FEATURES])
    cats=list(range(len(NUMERIC),len(FEATURES)))
    categorical_names={j:levels[CATEGORICAL[j-len(NUMERIC)]] for j in cats}
    explainer=LimeTabularExplainer(tr,feature_names=FEATURES,class_names=['Permanece','Abandona'],
                                   categorical_features=cats,categorical_names=categorical_names,
                                   mode='classification',discretize_continuous=True,random_state=SEED)
    return explainer,predict_numeric,pt


def lime_for_new_customer(pipe,X_train,X_new,n_features=10,num_samples=850):
    """LIME local en vivo sobre 19 variables originales; el modelo no se reentrena."""
    exp, predict_numeric, rows=build_lime_explainer(pipe,X_train,X_new)
    lime=exp.explain_instance(rows[0],predict_numeric,labels=[1],num_features=n_features,
                              num_samples=num_samples)
    return pd.DataFrame(lime.as_list(label=1),columns=['Regla','Peso hacia abandono'])
