"""Genera un pitch comercial FaroCX de 5 diapositivas en LaTeX Beamer.
Usa únicamente artefactos de evaluación existentes; no entrena ni modifica modelos.
"""
from __future__ import annotations
import json
import shutil
import subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'
OUT_TEX = ROOT / 'Pitch_FaroCX_Kevin_Erazo.tex'
OUT_PDF = ROOT / 'Pitch_FaroCX_Kevin_Erazo.pdf'


def escape_latex(value):
    chars = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
             '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
             '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(chars.get(ch, ch) for ch in str(value))


def load_data():
    required = ['metricas_test_final.json', 'comparacion_optimizadores_cv.csv',
                'seleccion_modelo_final.json', 'eda_auditoria.json', 'shap_global.csv']
    missing = [n for n in required if not (ART/n).exists()]
    if missing:
        raise FileNotFoundError('Faltan artefactos ya entrenados: ' + ', '.join(missing))
    metrics = json.loads((ART/'metricas_test_final.json').read_text(encoding='utf-8'))
    optimizers = pd.read_csv(ART/'comparacion_optimizadores_cv.csv')
    selected = json.loads((ART/'seleccion_modelo_final.json').read_text(encoding='utf-8'))
    audit = json.loads((ART/'eda_auditoria.json').read_text(encoding='utf-8'))
    shap = pd.read_csv(ART/'shap_global.csv')
    return metrics, optimizers, selected, audit, shap


def make_source(metrics, optimizers, selected, audit, shap):
    final = metrics['modelo_final']
    lr = metrics['logistic_regression_baseline']
    random = optimizers[optimizers['estrategia'].eq('Random Search')].iloc[0]
    hyper = optimizers[optimizers['estrategia'].str.startswith('Hyperband')].iloc[0]
    if abs(float(random.f1_cv)-float(hyper.f1_cv)) > 0.00005:
        comparison = (f"Random Search: {float(random.f1_cv):.4f} F1-CV; "
                      f"Hyperband: {float(hyper.f1_cv):.4f} F1-CV.")
    else:
        faster = 'Random Search' if float(random.tiempo_busqueda_s)<float(hyper.tiempo_busqueda_s) else 'Hyperband'
        comparison = f"Random Search y Hyperband: F1-CV {float(random.f1_cv):.4f}; {faster} fue más rápido en esta ejecución."
    n = int(audit['rows']); rate=float(audit['churn_rate']); churn=int(audit['churn_yes'])
    main_shap = ', '.join(escape_latex(s) for s in shap.head(3)['Variable'])
    context = {
        '__N__':f'{n:,}'.replace(',','.'), '__RATE__':f'{rate*100:.1f}'.replace('.',','),
        '__CHURN__':f'{churn:,}'.replace(',','.'),
        '__F1__':f'{final["f1"]:.3f}', '__AUC__':f'{final["roc_auc"]:.3f}',
        '__LR_F1__':f'{lr["f1"]:.3f}', '__LR_AUC__':f'{lr["roc_auc"]:.3f}',
        '__CV__':f'{selected["f1_cv"]:.4f}',
        '__COMP__':escape_latex(comparison), '__SHAP__':main_shap,
    }
    template = r"""\documentclass[aspectratio=169,10pt]{beamer}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[spanish,es-noshorthands]{babel}
\usepackage{lmodern}
\usepackage{tikz}
\usepackage{array,booktabs,tabularx}
\usetikzlibrary{positioning,arrows.meta}
\definecolor{ink}{HTML}{142F3D}
\definecolor{teal}{HTML}{087C78}
\definecolor{mint}{HTML}{EAF7F2}
\definecolor{gold}{HTML}{E49A3F}
\definecolor{light}{HTML}{F4F7F8}
\definecolor{muted}{HTML}{536976}
\setbeamercolor{normal text}{fg=ink,bg=white}
\setbeamercolor{frametitle}{fg=ink,bg=white}
\setbeamerfont{frametitle}{size=\Large,series=\bfseries}
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{headline}{}
\setbeamertemplate{footline}{\leavevmode\hbox{\begin{beamercolorbox}[wd=.88\paperwidth,ht=3.4ex,dp=1ex,leftskip=.6cm]{normal text}\scriptsize\color{teal}\textbf{FARO\textit{CX}}\quad\color{muted} | Intelligence for retention \qquad\color{muted} Kevin Andrés Erazo Yepez\end{beamercolorbox}\begin{beamercolorbox}[wd=.12\paperwidth,ht=3.4ex,dp=1ex,center]{normal text}\scriptsize\color{teal}\insertframenumber\,/\,5\end{beamercolorbox}}}
\newcommand{\eyebrow}[1]{{\vspace{-.15cm}\scriptsize\bfseries\color{teal}\MakeUppercase{#1}\par\vspace{.18cm}}}
\newcommand{\kpi}[3]{\begin{minipage}{.29\linewidth}\colorbox{mint}{\parbox{\dimexpr\linewidth-2\fboxsep}{\centering\vspace{.16cm}\LARGE\bfseries\color{teal}#1\par\vspace{.09cm}{\small\color{ink}#2}\vspace{.16cm}}}\par\vspace{.08cm}{\scriptsize\color{muted}#3}\end{minipage}}
\newcommand{\card}[2]{\colorbox{light}{\parbox{.90\linewidth}{\vspace{.12cm}\textbf{\color{teal}#1}\par\vspace{.09cm}{\small\color{ink}#2}\vspace{.14cm}}}}
\begin{document}

% 1. PROBLEMA Y POSICIONAMIENTO
\begin{frame}[t]{La fuga se anticipa. La decisión también.}
\eyebrow{01 / Oportunidad comercial}
\vspace{.07cm}
{\large En una operadora, cada contacto cuesta. Actuar a ciegas desperdicia presupuesto; no actuar deja pasar oportunidades de retención.}\par
\vspace{.28cm}
\kpi{__N__}{clientes}{Cartera histórica Telco}\hfill
\kpi{__RATE__\%}{abandono observado}{No es una predicción futura}\hfill
\kpi{__CHURN__}{bajas registradas}{Base pública de demostración}
\vspace{.40cm}
\colorbox{ink}{\parbox{.96\linewidth}{\vspace{.19cm}\centering\large\bfseries\color{white}No preguntamos solo quién podría irse.\\[.05cm]Preguntamos: ¿en quién conviene invertir primero?\vspace{.19cm}}}
\vspace{.21cm}
{\scriptsize\color{muted}FaroCX convierte un pronóstico estadístico en un escenario de priorización comercial; no garantiza que una intervención evite una baja.}
\end{frame}

% 2. LA SOLUCION
\begin{frame}[t]{FaroCX: riesgo, explicación y acción en un solo flujo}
\eyebrow{02 / Propuesta diferencial}
\vspace{.10cm}
\begin{columns}[T,onlytextwidth]
\column{.31\textwidth}
\card{01 \quad DETECTAR}{Estima el riesgo individual con un modelo evaluado sobre datos independientes.}
\vspace{.20cm}
\card{02 \quad ENTENDER}{SHAP y LIME muestran qué señales influyen en cada pronóstico.}
\column{.31\textwidth}
\card{03 \quad PRIORIZAR}{Ordena candidatos según el beneficio neto esperado y el presupuesto disponible.}
\vspace{.20cm}
\card{04 \quad MEDIR}{Propone validar el impacto de las campañas con un piloto controlado A/B.}
\column{.32\textwidth}
{\color{teal}\bfseries El valor no es una lista de porcentajes.}\par
\vspace{.18cm}
{\large Es una \textbf{lista de decisiones justificables}.}\par
\vspace{.22cm}
{\small\color{muted}Riesgo $\times$ eficacia supuesta $\times$ margen estimado $-$ costo de contacto.}
\end{columns}
\vspace{.32cm}
{\small\color{teal}\textbf{Diferenciación:} el equipo de CRM puede variar presupuesto y costos para comparar escenarios antes de contactar.}
\end{frame}

% 3. DEMO
\begin{frame}[t]{Una demostración que termina en una decisión}
\eyebrow{03 / Producto funcionando en la nube}
\vspace{.10cm}
{\large Tres pantallas; una pregunta de negocio: \textbf{¿a quién contactamos primero?}}\par
\vspace{.28cm}
\begin{columns}[T,onlytextwidth]
\column{.32\textwidth}
\card{PERFIL}{Introducir datos de un cliente y consultar el riesgo estimado de abandono.}
\column{.32\textwidth}
\card{EXPLICACIÓN}{Mostrar factores locales SHAP y LIME: por qué el modelo emitió esa alerta.}
\column{.32\textwidth}
\card{CAMPAÑA}{Ajustar presupuesto, margen y costo de contacto; exportar una cartera priorizada.}
\end{columns}
\vspace{.40cm}
\colorbox{mint}{\parbox{.96\linewidth}{\vspace{.16cm}\centering\bfseries\color{ink}FaroCX hace visible el paso de \textcolor{teal}{dato} a \textcolor{teal}{decisión}.\vspace{.16cm}}}
\vspace{.24cm}
{\scriptsize\color{muted}Los escenarios económicos son hipotéticos. Esta versión usa datos públicos de demostración y no integra información privada de clientes reales.}
\end{frame}

% 4. EVIDENCIA
\begin{frame}[t]{Evidencia comprobable, sin promesas infladas}
\eyebrow{04 / Validación e interpretabilidad}
\vspace{.03cm}
\begin{columns}[T,onlytextwidth]
\column{.50\textwidth}
{\small\bfseries Rendimiento en test reservado}\par\vspace{.08cm}
\begin{tabular}{@{}lcc@{}}\toprule
\small Modelo & \small F1 & \small ROC-AUC\\\midrule
\small Logistic Regression & \textbf{__LR_F1__} & \textbf{__LR_AUC__}\\
\small RF seleccionado por NSGA-II & __F1__ & __AUC__\\\bottomrule
\end{tabular}
\vspace{.17cm}
{\scriptsize\color{muted}Logistic Regression obtuvo mejores métricas en test. El Random Forest fue seleccionado antes mediante criterios de F1-CV y latencia (F1-CV=__CV__).}
\column{.46\textwidth}
{\small\bfseries Modelos explicables para CRM}\par\vspace{.09cm}
{\small SHAP global destaca: \textbf{__SHAP__}.}\par\vspace{.15cm}
{\small LIME y SHAP explican el perfil individual.}\par\vspace{.15cm}
{\small Grid Search, Random Search, Hyperband y NSGA-II documentan el proceso de optimización.}
\end{columns}
\vspace{.32cm}
\colorbox{light}{\parbox{.96\linewidth}{\vspace{.12cm}\scriptsize\color{muted}__COMP__\par\vspace{.06cm}Ni las explicaciones ni las simulaciones establecen causalidad ni efectividad comercial.\vspace{.12cm}}}
\end{frame}

% 5. NEGOCIACION
\begin{frame}[t]{No le pido que compre una promesa. Le propongo medirla.}
\eyebrow{05 / Propuesta comercial}
\vspace{.08cm}
\begin{columns}[T,onlytextwidth]
\column{.48\textwidth}
{\large\bfseries Piloto acotado con su equipo de CRM}\par\vspace{.18cm}
\card{ALCANCE}{Cartera autorizada, perfiles de riesgo, explicaciones y escenarios de contacto.}
\vspace{.16cm}
\card{MODELO COMERCIAL}{Referencia inicial: USD 59/mes (hipótesis ajustable al alcance).}
\column{.48\textwidth}
{\large\bfseries Criterios de continuidad}\par\vspace{.18cm}
\begin{enumerate}\small
\item Calidad predictiva sobre datos del cliente.
\item Costo por contacto y adopción por CRM.
\item Bajas incrementales evitadas, medidas con control A/B.
\item Privacidad, sesgos y seguridad revisados.
\end{enumerate}
\end{columns}
\vspace{.33cm}
\colorbox{ink}{\parbox{.96\linewidth}{\vspace{.21cm}\centering\large\bfseries\color{white}¿Definimos un piloto con presupuesto, métricas y fecha de revisión?\vspace{.21cm}}}
\vspace{.13cm}
{\scriptsize\color{muted}Se necesita un horizonte adecuado para observar bajas; no se afirma ROI ni reducción de churn demostrados.}
\end{frame}
\end{document}
"""
    for k,v in context.items():
        template=template.replace(k,v)
    return template


def main():
    data=load_data()
    OUT_TEX.write_text(make_source(*data),encoding='utf-8')
    print('Fuente LaTeX generada:', OUT_TEX)
    if not shutil.which('pdflatex'):
        print('pdflatex no está disponible; PDF no compilado')
        return
    cmd=['pdflatex','-interaction=nonstopmode','-halt-on-error','-file-line-error',
         '-output-directory',str(ROOT),str(OUT_TEX)]
    for _ in range(2):
        proc=subprocess.run(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        if proc.returncode:
            print(proc.stdout[-5000:])
            raise RuntimeError('No se pudo compilar LaTeX')
    for ext in ('aux','log','nav','out','snm','toc'):
        p=ROOT/f'{OUT_TEX.stem}.{ext}'
        if p.exists():p.unlink()
    try:
        import fitz
        assert len(fitz.open(OUT_PDF))==5, 'El PDF debe tener 5 páginas'
    except ImportError:
        pass
    print('PDF generado (5 diapositivas):', OUT_PDF)

if __name__=='__main__':
    main()
