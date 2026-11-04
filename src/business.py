"""Escenarios económicos HIPOTÉTICOS, no efectos causales estimados."""
import numpy as np
import pandas as pd


def simulate_retention(clients:pd.DataFrame, intervention_cost=8.0,success_rate=.15,
                       gross_margin_rate=.40,horizon_months=6,budget=500.0,
                       saas_monthly_fee=59.0):
    """Se priorizan clientes por beneficio neto esperado p*eficacia*margen*horizonte-costo.

    El efecto de la campaña, margen y costo son supuestos editables, no medidos en Telco.
    La probabilidad predicha es asociativa, no el resultado de una intervención.
    """
    d=clients.copy()
    d['margen_horizonte_hipotetico']=d['MonthlyCharges']*gross_margin_rate*horizon_months
    d['beneficio_bruto_esperado']=d['probabilidad_churn']*success_rate*d['margen_horizonte_hipotetico']
    d['beneficio_neto_esperado']=d['beneficio_bruto_esperado']-intervention_cost
    d=d.sort_values('beneficio_neto_esperado',ascending=False)
    number=min(int(max(0,budget)//intervention_cost) if intervention_cost>0 else len(d),len(d))
    selected=d.loc[d['beneficio_neto_esperado']>0].head(number)
    spend=intervention_cost*len(selected)
    gross_gain=float(selected['beneficio_bruto_esperado'].sum())
    net=gross_gain-spend-saas_monthly_fee
    denom=float((selected['probabilidad_churn']*selected['margen_horizonte_hipotetico']).sum())
    breakeven=float((spend+saas_monthly_fee)/denom) if denom>0 else None
    scenario={'contactados':int(len(selected)), 'costo_acciones_usd':float(spend),
              'costo_software_usd':float(saas_monthly_fee),
              'beneficio_bruto_esperado_usd':float(gross_gain),
              'resultado_neto_hipotetico_usd':float(net),
              'roi_hipotetico':float(net/(spend+saas_monthly_fee)) if (spend+saas_monthly_fee)>0 else 0.,
              'eficacia_minima_break_even':breakeven,
              'advertencia':'Supuestos no causales: sin ensayo A/B NO hay evidencia de uplift ni retorno real.'}
    return selected,scenario
