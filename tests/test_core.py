import numpy as np
import pandas as pd
from src.core import NUMERIC,CATEGORICAL,FEATURES,make_preprocessor,split_data,format_user_input
from src.business import simulate_retention
from src.experiments import rf_pipe,lr_pipe


def test_feature_count():
    assert len(NUMERIC)==3 and len(CATEGORICAL)==16 and len(FEATURES)==19


def test_preprocess_synthetic_schema_only():
    a={n:['No','Yes','No','Yes','No','Yes'] for n in CATEGORICAL}
    a['gender']=['Male','Female','Male','Female','Male','Female']
    a['SeniorCitizen']=['No','Yes']*3
    a.update({'tenure':[1,2,3,4,5,6],'MonthlyCharges':[30.,40.,50.,60.,70.,80.],
              'TotalCharges':[30.,80.,150.,240.,350.,480.]})
    x=pd.DataFrame(a)[FEATURES]
    p=make_preprocessor().fit(x)
    assert p.transform(x).shape[0]==len(x)
    assert np.isfinite(p.transform(x)).all()


def test_business_assumptions():
    d=pd.DataFrame({'customerID':['a','b'],'MonthlyCharges':[100.,50.],
                    'probabilidad_churn':[.9,.1]})
    picked,s=simulate_retention(d,intervention_cost=8,success_rate=.15,gross_margin_rate=.4,
                                horizon_months=6,budget=50,saas_monthly_fee=59)
    assert len(picked)<=2 and 'resultado_neto_hipotetico_usd' in s
