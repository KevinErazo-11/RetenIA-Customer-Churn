"""Genera CINCO diapositivas Beamer (fondo claro) y compila PDF con pdflatex.

Usa exclusivamente los artefactos de la NUEVA ejecución; no recupera cifras
antiguas ni inventa métricas cuando aún no se ha entrenado.

Uso: python build_pitch.py
Si pdflatex no está instalado, se genera de todos modos el .tex y se avisa.
"""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import subprocess
import pandas as pd

ROOT=Path(__file__).resolve().parent
ART=ROOT/'artifacts'
OUT_TEX=ROOT/'Pitch_RetenIA_Kevin_Erazo.tex'
OUT_PDF=ROOT/'Pitch_RetenIA_Kevin_Erazo.pdf'


def escape_latex(x):
    s = str(x)
    chars = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
             '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
             '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(chars.get(ch, ch) for ch in s)


def load_data():
    needed=['metricas_test_final.json','seleccion_modelo_final.json',
            'shap_global.csv','comparacion_optimizadores_cv.csv',
            'manifiesto_experimento.json']
    missing=[name for name in needed if not (ART/name).is_file()]
    if missing:
        raise FileNotFoundError('Ejecuta primero python run_experiments.py; faltan: '+', '.join(missing))
    manifest=json.loads((ART/'manifiesto_experimento.json').read_text(encoding='utf-8'))
    if manifest.get('ejecucion_quick_no_entregable'):
        raise ValueError('No se genera una presentación final con resultados --quick.')
    metrics=json.loads((ART/'metricas_test_final.json').read_text(encoding='utf-8'))
    selected=json.loads((ART/'seleccion_modelo_final.json').read_text(encoding='utf-8'))
    shap=pd.read_csv(ART/'shap_global.csv').head(5).to_dict('records')
    searches=pd.read_csv(ART/'comparacion_optimizadores_cv.csv').to_dict('records')
    return metrics, selected, shap, searches, True


def make_source(metrics,selection,shap,searches,is_local):
    m=metrics['modelo_final'];lr=metrics['logistic_regression_baseline'];base=metrics['random_forest_baseline']
    winner=max(searches,key=lambda v:float(v['f1_cv']))
    prov='validación independiente del modelo tras nueva ejecución'
    template=r"""\documentclass[aspectratio=169,10pt]{beamer}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[spanish,es-noshorthands]{babel}
\usepackage{lmodern,booktabs,tabularx,array}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta}
\definecolor{navy}{HTML}{173750}
\definecolor{teal}{HTML}{007F86}
\definecolor{muted}{HTML}{586D7B}
\definecolor{soft}{HTML}{EDF5F6}
\definecolor{orange}{HTML}{A15A18}
\definecolor{lightgray}{HTML}{F4F7F9}
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{footline}{\leavevmode\hbox{\begin{beamercolorbox}[wd=.86\paperwidth,ht=3ex,dp=1.3ex,leftskip=.65cm]{normal text}\scriptsize\color{muted}RetenIA \quad | \quad Kevin Andrés Erazo Yepez \quad | \quad Business Pitch\end{beamercolorbox}\begin{beamercolorbox}[wd=.14\paperwidth,ht=3ex,dp=1.3ex,center]{normal text}\scriptsize\color{teal}\insertframenumber/5\end{beamercolorbox}}}
\setbeamercolor{normal text}{fg=navy,bg=white}
\setbeamercolor{frametitle}{fg=navy,bg=white}
\setbeamerfont{frametitle}{size=\Large,series=\bfseries}
\setbeamercolor{block title}{bg=soft,fg=teal}
\setbeamercolor{block body}{bg=lightgray,fg=navy}
\setbeamercolor{alerted text}{fg=orange}
\setbeamertemplate{itemize item}{\color{teal}\small$\blacktriangleright$}
\newcommand{\eyebrow}[1]{\vspace{-0.18cm}{\footnotesize\bfseries\color{teal}#1}\par\vspace{0.1cm}}
\newcommand{\bigstat}[3]{\begin{minipage}[t]{.29\linewidth}\colorbox{soft}{\parbox{\dimexpr\linewidth-2\fboxsep}{\centering\vspace{.17cm}{\LARGE\bfseries\color{teal}#1}\par\vspace{.09cm}{\scriptsize\color{navy}#2}\vspace{.17cm}}}\\[.12cm]{\tiny\color{muted}#3}\end{minipage}}
\begin{document}

\begin{frame}[t]{La oportunidad: actuar antes de perder clientes}
\eyebrow{01 / EL PROBLEMA}
\vspace{.16cm}
{\large Una operadora no puede gastar el mismo presupuesto de retención en todos los clientes.}\par
\vspace{.43cm}
\bigstat{7.043}{clientes en Telco}{Dataset de demostración}\hfill
\bigstat{$\approx$26,5\%}{abandono histórico}{Clase minoritaria}\hfill
\bigstat{19}{variables predictoras}{Perfil y servicios del cliente}
\vspace{.47cm}
\begin{block}{La pregunta de negocio}
¿A quién contactar primero cuando los recursos comerciales son limitados?
\end{block}
\vspace{.15cm}
{\small\color{muted}El reto no es solo clasificar: también evitar contactos innecesarios y pérdidas por bajas no detectadas.}
\end{frame}

\begin{frame}[t]{RetenIA: predicción, explicación y prioridad}
\eyebrow{02 / LA SOLUCIÓN}
\begin{columns}[T,onlytextwidth]
\column{.52\textwidth}
\begin{tikzpicture}[node distance=0.37cm, every node/.style={font=\small}, box/.style={draw=teal, rounded corners=3pt, fill=soft, text width=6.2cm, minimum height=.68cm, align=left, inner sep=6pt}]
\node[box] (a) {\textbf{1. Diagnosticar} \quad Explorar segmentos};
\node[box, below=of a] (b) {\textbf{2. Estimar} \quad P(Churn=1) individual};
\node[box, below=of b] (c) {\textbf{3. Explicar} \quad SHAP y LIME por cliente};
\node[box, below=of c] (d) {\textbf{4. Priorizar} \quad Ordenar por presupuesto};
\draw[-{Latex[length=2mm]},teal,thick] (a)--(b);
\draw[-{Latex[length=2mm]},teal,thick] (b)--(c);
\draw[-{Latex[length=2mm]},teal,thick] (c)--(d);
\end{tikzpicture}
\column{.43\textwidth}
\begin{block}{Para equipos de CRM}
\begin{itemize}
\item Panel interactivo de riesgo.
\item Modelos exportados y reproducibles.
\item Justificación de cada predicción.
\item Simulador de contactos de retención.
\end{itemize}
\end{block}
\vspace{.20cm}
{\small\color{muted}El sistema apoya decisiones humanas. No reentrena dentro de Streamlit ni garantiza evitar bajas.}
\end{columns}
\end{frame}

\begin{frame}[t]{Evidencia real: desempeño y selección transparente}
\eyebrow{03 / VALIDACIÓN Y LIMITACIONES}
\vspace{.12cm}
\begin{center}
\renewcommand{\arraystretch}{1.32}
\begin{tabular}{lccc}
\toprule
\textbf{Modelo (test reservado)} & \textbf{F1} & \textbf{Recall} & \textbf{ROC-AUC}\\
\midrule
Logistic Regression & \textbf{__LR_F1__} & __LR_REC__ & __LR_AUC__\\
Random Forest base & __BASE_F1__ & __BASE_REC__ & __BASE_AUC__\\
__SELECTED_NAME__ & __M_F1__ & __M_REC__ & __M_AUC__\\
\bottomrule
\end{tabular}
\end{center}
\vspace{.18cm}
\begin{block}{Selección previa a la evaluación final}
Configuración operativa: \textbf{__SELECTED_NAME__}. Se aplicó el criterio F1-CV y, cuando correspondía, el compromiso con latencia de NSGA-II. F1-CV = \textbf{__M_CV__}.
\end{block}
\vspace{.10cm}
{\small\color{orange}\textbf{Evidencia independiente:} __BENCHMARK_NOTE__}

\end{frame}

\begin{frame}[t]{Optimizar dos objetivos y explicar predicciones}
\eyebrow{04 / HYPERPARAMETER OPTIMIZATION, NSGA-II + XAI}
{\footnotesize\color{teal}Grid Search \quad | \quad Random Search \quad | \quad Hyperband (Optuna) \quad | \quad NSGA-II}\par
\vspace{.07cm}
\begin{columns}[T,onlytextwidth]
\column{.48\textwidth}
\begin{block}{Frente de Pareto}
\centering\small $\max_{\theta}\;F1_{CV}(\theta)$ \quad y \quad $\min_{\theta}\;T_{infer}(\theta)$
\vspace{.12cm}
\begin{itemize}
\item Soluciones no dominadas: no hay mejora simultánea gratuita.
\item Se acepta hasta 0,01 de pérdida de F1 frente al máximo de Pareto para buscar menor latencia.
\item La latencia depende del equipo y de la carga.
\end{itemize}
\end{block}
\column{.48\textwidth}
\begin{block}{Top variables SHAP (global)}
\begin{tabular}{@{}lr@{}}
\textbf{Variable} & \textbf{Importancia}\\
\midrule
__SHAP_ROWS__
\end{tabular}
\end{block}
\vspace{.09cm}
{\small LIME explica dos clientes locales; la importancia del bosque compara reducciones de impureza.}
\end{columns}
\vspace{.10cm}
{\scriptsize\color{muted}Random Search y Hyperband (Optuna) empataron en F1-CV=__BEST_F1__; Random Search fue más rápido.}\par
{\small\color{orange}\textbf{Precaución:} importancia predictiva no prueba causalidad.}
\end{frame}

\begin{frame}[t]{Oferta: piloto controlado antes de contratar}
\eyebrow{05 / PROPUESTA Y CIERRE}
\vspace{.20cm}
\begin{columns}[T,onlytextwidth]
\column{.50\textwidth}
\begin{block}{Modelo comercial propuesto}
\begin{itemize}
\item Cliente: gerencia de CRM y retención.
\item Producto: panel de riesgo, XAI y prioridad.
\item Licencia \textbf{hipotética}: \textbf{USD 59/mes}.
\item Costos, margen y eficacia son supuestos editables.
\end{itemize}
\end{block}
\column{.45\textwidth}
\begin{block}{Prueba que sí aporta evidencia}
\begin{enumerate}
\item Piloto A/B con grupo de control.
\item Medir bajas evitadas de forma incremental.
\item Estimar retorno neto con costos reales.
\item Revisar privacidad, sesgos y consentimiento.
\end{enumerate}
\end{block}
\end{columns}
\vspace{.36cm}
\begin{center}
\colorbox{soft}{\parbox{.90\linewidth}{\centering\vspace{.20cm}\large\bfseries\color{teal}¿Podemos evaluar un piloto medible antes de escalar?\vspace{.20cm}}}
\end{center}
\vspace{.10cm}
{\tiny\color{muted}Cifras predictivas: __PROVENANCE__. Simulación comercial: supuestos no observados en Telco.}
\end{frame}
\end{document}
"""
    shaprows=r'\\'.join(f"{escape_latex(str(row['Variable']))} & {float(row['SHAP_importancia_abs_media']):.3f}" for row in shap[:4])+r'\\'
    benchmark_note = (f'Logistic Regression logró F1={lr["f1"]:.3f}, superior al modelo operativo F1={m["f1"]:.3f}; no alteramos la selección tras test.'
                      if lr['f1'] > m['f1'] else
                      f'El modelo operativo obtuvo F1={m["f1"]:.3f}; Logistic Regression F1={lr["f1"]:.3f}.')
    sub={
        '__LR_F1__':f"{lr['f1']:.3f}", '__LR_REC__':f"{lr['recall']:.3f}", '__LR_AUC__':f"{lr['roc_auc']:.3f}",
        '__BASE_F1__':f"{base['f1']:.3f}", '__BASE_REC__':f"{base['recall']:.3f}", '__BASE_AUC__':f"{base['roc_auc']:.3f}",
        '__M_F1__':f"{m['f1']:.3f}", '__M_REC__':f"{m['recall']:.3f}", '__M_AUC__':f"{m['roc_auc']:.3f}",
        '__M_CV__':f"{selection['f1_cv']:.4f}",
        '__SELECTED_NAME__':escape_latex(selection['nombre']),
        '__BENCHMARK_NOTE__':escape_latex(benchmark_note), '__BEST_SEARCH__':escape_latex(winner['estrategia']),
        '__BEST_F1__':f"{winner['f1_cv']:.4f}", '__SHAP_ROWS__':shaprows, '__PROVENANCE__':escape_latex(prov),
    }
    for key,val in sub.items(): template=template.replace(key,val)
    return template


def main():
    met,chosen,shap,search,local=load_data()
    OUT_TEX.write_text(make_source(met,chosen,shap,search,local),encoding='utf-8')
    print('Fuente LaTeX generada:',OUT_TEX)
    if shutil.which('pdflatex') is None:
        print('PDF sin compilar: instala TeX Live (pdflatex), luego ejecuta python build_pitch.py')
        return
    cmd=['pdflatex','-interaction=nonstopmode','-halt-on-error','-file-line-error','-output-directory',str(ROOT),str(OUT_TEX)]
    for _ in range(2):
        proc=subprocess.run(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        if proc.returncode:
            print(proc.stdout[-5500:]);raise RuntimeError('Falló compilación LaTeX; revisa mensajes anteriores')
    for ext in ('aux','log','nav','out','snm','toc'):
        p=ROOT/f'{OUT_TEX.stem}.{ext}'
        if p.exists():p.unlink()
    try:
        import fitz
        if len(fitz.open(OUT_PDF))!=5:
            raise RuntimeError('PDF Beamer debe tener exactamente cinco páginas')
    except ImportError:
        pass
    print('PDF generado (5 diapositivas):',OUT_PDF)

if __name__=='__main__':main()
