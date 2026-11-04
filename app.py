"""RetenIA · Plataforma de inteligencia de retención de clientes.

Interfaz comercial. El análisis, el rendimiento, las explicaciones y los escenarios
se derivan de los artefactos del proyecto. No se realizan ajustes del modelo aquí.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.business import simulate_retention
from src.core import (
    CATEGORICAL,
    FEATURES,
    NUMERIC,
    PROJECT_ROOT,
    format_user_input,
    load_telco,
)

LOGGER = logging.getLogger(__name__)
ART = PROJECT_ROOT / "artifacts"
FIG = PROJECT_ROOT / "figures"

st.set_page_config(
    page_title="RetenIA | Inteligencia de retención",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1320px; padding-top: 1.35rem; padding-bottom: 3.5rem;}
    [data-testid="stSidebar"] {border-right: 1px solid rgba(135,165,186,.16);}
    [data-testid="stMetric"] {
        border: 1px solid rgba(115,160,175,.22);
        background: rgba(55,88,111,.09);
        border-radius: 14px; padding: 1rem 1.15rem;
    }
    [data-testid="stMetricLabel"] {font-weight: 500;}
    h1,h2,h3 {letter-spacing: -.025em;}
    div[data-testid="stAlert"] {border-radius: 12px;}
    [data-testid="stDataFrame"] {border-radius: 12px;}
    .rt-eyebrow {font-size:.77rem; font-weight:700; letter-spacing:.13em;
      text-transform:uppercase; color:#54D2C6; margin-bottom:.2rem;}
    .rt-lead {font-size:1.08rem; color:#AFC0CE; line-height:1.6;
      max-width:860px; margin-top:.2rem; margin-bottom:1.2rem;}
    .rt-panel {padding:1.1rem 1.25rem; border-radius:14px;
      border:1px solid rgba(115,160,175,.2);
      background:linear-gradient(120deg,rgba(25,73,80,.22),rgba(28,44,68,.18));}
    .rt-panel p {margin:0;line-height:1.7;}
    .rt-small {color:#9FB1C1;font-size:.84rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

PAGES = [
    "Inicio",
    "Exploratory Data Analysis",
    "Model Performance",
    "Hyperparameter Optimization",
    "NSGA-II · Pareto Frontier",
    "Explainable AI · SHAP y LIME",
    "Prediction & Explainability",
    "Business Application",
    "Información del servicio",
]
PAGE_ICONS = ["⌂", "◫", "◉", "⌁", "◇", "✦", "◎", "▤", "ⓘ"]

st.sidebar.markdown("# 📡 RetenIA")
st.sidebar.caption("Inteligencia para la retención de clientes")
st.sidebar.divider()
st.sidebar.markdown("**PLATAFORMA**")
selected = st.sidebar.radio(
    "Módulo",
    PAGES,
    format_func=lambda p: f"{PAGE_ICONS[PAGES.index(p)]}  {p}",
    label_visibility="collapsed",
    key="rt_page",
)
st.sidebar.divider()
st.sidebar.caption("Decisiones respaldadas por datos · Análisis de riesgo y priorización de cartera")


@st.cache_resource(show_spinner=False)
def model():
    return joblib.load(ART / "modelo_final_pipeline.joblib")


@st.cache_resource(show_spinner=False)
def xai_background():
    return joblib.load(ART / "xai_background_train.joblib")


@st.cache_data(show_spinner=False)
def dataset():
    df, _ = load_telco(download_if_missing=True)
    return df


@st.cache_data(show_spinner=False)
def csv_artifact(name: str) -> pd.DataFrame:
    return pd.read_csv(ART / name)


def artifact_json(name: str, default=None):
    path = ART / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def missing_data():
    required = [
        "modelo_final_pipeline.joblib",
        "metricas_test_final.json",
        "comparacion_optimizadores_cv.csv",
        "frente_pareto.csv",
    ]
    return [name for name in required if not (ART / name).is_file()]


def page_title(eyebrow: str, title: str, summary: str):
    st.markdown(f'<div class="rt-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="rt-lead">{summary}</div>', unsafe_allow_html=True)


def insight(message: str):
    st.markdown(f'<div class="rt-panel"><p>{message}</p></div>', unsafe_allow_html=True)


def show_chart(fig, key=None):
    fig.update_layout(
        template="plotly_dark",
        font=dict(size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=55, b=24),
        legend_title_text="",
    )
    st.plotly_chart(fig, width="stretch", key=key)


def saved_image(name: str, caption: str = ""):
    source = FIG / name
    if not source.is_file():
        source = ART / name
    if source.is_file():
        st.image(str(source), caption=caption or None, width="stretch")
    else:
        st.caption("Esta visualización no está disponible en la versión publicada.")


def optional_csv(name: str):
    return csv_artifact(name) if (ART / name).is_file() else None


def short_metrics(data: dict, prefix: str = ""):
    metric_labels = [("accuracy", "Exactitud"), ("precision", "Precisión"),
                     ("recall", "Detección de bajas"), ("f1", "F1-score"),
                     ("roc_auc", "ROC-AUC")]
    cols = st.columns(5)
    for col, (key, label) in zip(cols, metric_labels):
        col.metric(prefix + label, f"{float(data.get(key, 0.0)):.3f}")


def download_table(data: pd.DataFrame, file_name: str, label: str):
    st.download_button(
        label,
        data.to_csv(index=False).encode("utf-8-sig"),
        file_name=file_name,
        mime="text/csv",
    )


TRANSLATIONS = {
    "Month-to-month": "Mensual", "One year": "Anual", "Two year": "Dos años",
    "Fiber optic": "Fibra óptica", "DSL": "DSL", "No": "No", "Yes": "Sí",
    "Female": "Mujer", "Male": "Hombre",
    "No internet service": "Sin servicio de Internet",
    "No phone service": "Sin servicio telefónico",
    "Electronic check": "Cheque electrónico",
    "Mailed check": "Cheque postal",
    "Bank transfer (automatic)": "Transferencia automática",
    "Credit card (automatic)": "Tarjeta de crédito automática",
}


def translate(value):
    return TRANSLATIONS.get(str(value), str(value))


def human_table(frame: pd.DataFrame, fmt: dict | None = None):
    st.dataframe(frame.style.format(fmt or {}), hide_index=True, width="stretch")


if missing_data():
    st.error("El servicio de análisis no está disponible temporalmente. Contacta al administrador de RetenIA.")
    LOGGER.error("Missing deployment assets: %s", missing_data())
    st.stop()

manifest = artifact_json("manifiesto_experimento.json", {})
if manifest.get("ejecucion_quick_no_entregable"):
    st.error("El servicio no dispone de un modelo validado para operar.")
    st.stop()

metrics = artifact_json("metricas_test_final.json", {})
selection = artifact_json("seleccion_modelo_final.json", {})

try:
    df = dataset()
except Exception:
    LOGGER.exception("Unable to load customer dataset")
    st.error("No fue posible cargar la información de clientes. Contacta al administrador del servicio.")
    st.stop()

# 01 · INICIO
if selected == "Inicio":
    page_title(
        "INTELIGENCIA COMERCIAL",
        "Decide dónde concentrar tus esfuerzos de retención",
        "RetenIA transforma los datos de clientes en señales de riesgo, explicaciones comprensibles y escenarios para orientar decisiones comerciales.",
    )
    a, b, c, d = st.columns(4)
    a.metric("Clientes analizados", f"{len(df):,}")
    b.metric("Tasa histórica de abandono", f"{df['Churn'].mean():.1%}")
    c.metric("Variables del perfil", f"{len(FEATURES)}")
    if metrics.get("modelo_final"):
        d.metric("Capacidad discriminativa · AUC", f"{metrics['modelo_final']['roc_auc']:.3f}")
    else:
        d.metric("Modelos disponibles", "1")

    st.subheader("De los datos a la acción")
    steps = [
        ("01", "Comprende tu cartera", "Explora segmentos, contratos, antigüedad y patrones de abandono."),
        ("02", "Valida la calidad", "Consulta métricas, alternativas de modelo y compromisos de rendimiento."),
        ("03", "Entiende el riesgo", "Revisa los factores más influyentes y sus explicaciones."),
        ("04", "Activa decisiones", "Evalúa clientes individuales y construye escenarios de contacto."),
    ]
    for row in [steps[:2], steps[2:]]:
        left, right = st.columns(2)
        for holder, (num, title, desc) in zip((left, right), row):
            with holder.container(border=True):
                st.caption(f"{num} / RETENIA")
                st.markdown(f"**{title}**")
                st.write(desc)
    st.divider()
    a, b = st.columns([1.25, 1])
    with a:
        st.subheader("Una visión más clara de las bajas")
        dist = df["Churn"].value_counts().reindex([0, 1], fill_value=0)
        chart = px.pie(
            values=dist.to_numpy(), names=["Permanecen", "Abandonan"], hole=.64,
            color_discrete_sequence=["#21B3AA", "#E1A568"], title="Distribución de clientes",
        )
        show_chart(chart)
    with b:
        st.subheader("Qué permite hacer RetenIA")
        st.markdown(
            "**Segmentar:** identificar grupos que presentan diferentes tasas de abandono.\n\n"
            "**Predecir:** estimar el riesgo individual para un perfil de cliente.\n\n"
            "**Explicar:** visualizar los atributos que influyen en la estimación.\n\n"
            "**Priorizar:** comparar posibles campañas bajo restricciones de presupuesto."
        )
        st.caption("Las predicciones son estimaciones estadísticas y no garantizan el comportamiento futuro de ningún cliente.")

# 02 · CLIENTES Y SEGMENTOS
elif selected == "Exploratory Data Analysis":
    page_title(
        "ANÁLISIS DE CARTERA", "Exploratory Data Analysis",
        "Explora el perfil de la cartera y compara el abandono histórico por contrato, antigüedad, servicios y facturación.",
    )
    f1, f2, f3 = st.columns(3)
    filters = {}
    for col, holder, label in (
        ("Contract", f1, "Tipo de contrato"),
        ("InternetService", f2, "Servicio de Internet"),
        ("PaymentMethod", f3, "Método de pago"),
    ):
        opts = ["Todos"] + sorted(df[col].dropna().astype(str).unique().tolist())
        filters[col] = holder.selectbox(label, opts, format_func=translate)
    segment = df.copy()
    for col, choice in filters.items():
        if choice != "Todos":
            segment = segment.loc[segment[col].astype(str).eq(choice)]
    if segment.empty:
        st.info("No hay clientes con esta combinación. Prueba con otros filtros.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Clientes del segmento", f"{len(segment):,}")
        b.metric("Abandono histórico", f"{segment['Churn'].mean():.1%}")
        c.metric("Cargo mensual promedio", f"USD {segment['MonthlyCharges'].mean():.2f}")
        d.metric("Antigüedad mediana", f"{segment['tenure'].median():.0f} meses")
        a, b = st.columns(2)
        by_contract = segment.groupby("Contract", dropna=False).agg(
            tasa=("Churn", "mean"), clientes=("Churn", "size")
        ).reset_index()
        by_contract["Contrato"] = by_contract["Contract"].map(translate)
        with a:
            fig = px.bar(by_contract, x="Contrato", y="tasa", text="clientes",
                         color="tasa", color_continuous_scale="Teal",
                         labels={"tasa": "Tasa de abandono", "clientes": "Clientes"},
                         title="Abandono por contrato")
            fig.update_yaxes(tickformat=".0%")
            show_chart(fig)
        with b:
            chartdata = segment.assign(Estado=segment["Churn"].map({0:"Permanece",1:"Abandona"}))
            show_chart(px.histogram(chartdata, x="tenure", color="Estado", nbins=24,
                                    barmode="overlay", title="Distribución de antigüedad",
                                    labels={"tenure":"Meses de permanencia"}))
        a, b = st.columns(2)
        with a:
            show_chart(px.box(chartdata, x="Estado", y="MonthlyCharges", color="Estado",
                              title="Cargo mensual por comportamiento",
                              labels={"MonthlyCharges":"USD / mes"}))
        by_internet = segment.groupby("InternetService", dropna=False).agg(
            tasa=("Churn", "mean"), clientes=("Churn", "size")
        ).reset_index()
        by_internet["Servicio"] = by_internet["InternetService"].map(translate)
        with b:
            fig = px.bar(by_internet, x="Servicio", y="tasa", text="clientes",
                         color="tasa", color_continuous_scale="Teal",
                         title="Abandono por servicio de Internet",
                         labels={"tasa":"Tasa de abandono", "clientes":"Clientes"})
            fig.update_yaxes(tickformat=".0%")
            show_chart(fig)
        st.caption("Las tasas describen asociaciones históricas; no indican que el contrato o el servicio causen la baja.")
    with st.expander("Calidad, cobertura y tratamiento de datos"):
        audit = artifact_json("eda_auditoria.json", {})
        a, b, c = st.columns(3)
        a.metric("Variables utilizadas", len(FEATURES))
        b.metric("Registros excluidos", audit.get("rows_dropped", 0))
        c.metric("Cargos acumulados corregidos", audit.get("totalcharges_filled_zero", "—"))
        st.dataframe(pd.DataFrame([
            ("Identificador del cliente", "Se utiliza para trazabilidad; no interviene en la predicción."),
            ("Cargos acumulados", "Se convierten a valores numéricos. Los casos nuevos sin historial se registran con cero."),
            ("Variables categóricas", "Se representan mediante codificación One-Hot."),
            ("Variables numéricas", "Se imputan y estandarizan dentro del proceso predictivo."),
            ("Evaluación", "Partición estratificada de entrenamiento y evaluación independiente."),
        ], columns=["Elemento", "Tratamiento"]), hide_index=True, width="stretch")
        st.caption("Se emplean tres variables numéricas y dieciséis categóricas. El preprocesamiento se aplica de forma consistente a las predicciones.")

# 03 · RENDIMIENTO
elif selected == "Model Performance":
    page_title(
        "CONFIANZA EN EL MODELO", "Model Performance",
        "Consulta la capacidad de identificación de posibles bajas y comprende los costos de las decisiones incorrectas.",
    )
    labels = [
        ("Logistic Regression", "logistic_regression_baseline"),
        ("Random Forest · baseline", "random_forest_baseline"),
        ("Random Forest · seleccionado", "modelo_final"),
    ]
    rows = []
    for label, key in labels:
        if key in metrics:
            values = metrics[key]
            rows.append({"Modelo":label, "Exactitud":float(values["accuracy"]),
                         "Precisión":float(values["precision"]),
                         "Recall":float(values["recall"]),
                         "F1":float(values["f1"]), "ROC-AUC":float(values["roc_auc"])})
    if rows:
        data = pd.DataFrame(rows)
        st.subheader("Comparación de alternativas")
        human_table(data, {k:"{:.3f}" for k in ["Exactitud","Precisión","Recall","F1","ROC-AUC"]})
        fig = px.bar(data, x="Modelo", y=["F1", "ROC-AUC"], barmode="group",
                     title="Calidad de predicción según el modelo",
                     labels={"value":"Puntuación", "variable":"Indicador"},
                     color_discrete_sequence=["#29B6AB", "#E3B36F"])
        fig.update_yaxes(range=[0,1])
        show_chart(fig)
    if "modelo_final" in metrics:
        st.subheader("Solución operativa")
        short_metrics(metrics["modelo_final"])
        lr = metrics.get("logistic_regression_baseline", {})
        final = metrics["modelo_final"]
        if lr and float(lr.get("f1",0)) > float(final["f1"]):
            st.caption(
                "Logistic Regression obtuvo un F1 superior en la evaluación independiente. "
                "La solución operativa se eligió considerando conjuntamente F1 en validación y tiempo de respuesta. "
                "Ambas alternativas deben compararse nuevamente antes de un despliegue productivo a gran escala."
            )
    st.subheader("Matriz de confusión")
    if "modelo_final" in metrics and "matriz_confusion" in metrics["modelo_final"]:
        matrix = np.asarray(metrics["modelo_final"]["matriz_confusion"], dtype=int)
        a, b, c, d = st.columns(4)
        a.metric("Permanencias bien identificadas", f"{matrix[0,0]:,}")
        b.metric("Alertas innecesarias", f"{matrix[0,1]:,}")
        c.metric("Bajas no detectadas", f"{matrix[1,0]:,}")
        d.metric("Bajas identificadas", f"{matrix[1,1]:,}")
        fig = px.imshow(
            matrix,
            x=["Predice permanencia", "Predice abandono"],
            y=["Permaneció", "Abandonó"],
            text_auto=True,
            color_continuous_scale="Teal",
            title="Resultados sobre clientes no usados para ajustar el modelo",
            labels={"x":"Pronóstico", "y":"Resultado real", "color":"Clientes"},
        )
        show_chart(fig)
        insight(
            "<strong>Cómo interpretar los errores.</strong> Una baja no detectada puede reducir la oportunidad de contacto preventivo. "
            "Una alerta innecesaria puede consumir presupuesto de retención. "
            "El equilibrio óptimo depende de los costos y de los objetivos del negocio."
        )
    st.subheader("Evaluación por umbral")
    a, b = st.columns(2)
    with a:
        saved_image("roc_test.png", "Curva ROC · discriminación para diferentes umbrales")
    with b:
        saved_image("pr_test.png", "Curva Precision–Recall · equilibrio entre alertas y detección")
    with st.expander("Metodología de validación"):
        baseline = optional_csv("comparacion_baselines_cv.csv")
        st.write("Las métricas de validación cruzada se obtienen con tres particiones estratificadas del conjunto de desarrollo. La evaluación final usa clientes separados de ese proceso.")
        if baseline is not None:
            st.dataframe(baseline.round(4), hide_index=True, width="stretch")
        st.caption("F1 es la media armónica de precisión y recall; ROC-AUC resume la capacidad discriminativa entre diferentes umbrales.")

# 04 · OPTIMIZACIÓN
elif selected == "Hyperparameter Optimization":
    page_title(
        "INGENIERÍA DE MODELOS", "Hyperparameter Optimization",
        "Comparación de estrategias para configurar un bosque aleatorio con atención a calidad de predicción y tiempo de búsqueda.",
    )
    search = csv_artifact("comparacion_optimizadores_cv.csv")
    display_columns = [c for c in ["estrategia", "f1_cv", "recall_cv", "roc_auc_cv", "tiempo_busqueda_s", "evaluaciones"] if c in search]
    display_names = {
        "estrategia":"Método", "f1_cv":"F1 de validación", "recall_cv":"Detección de bajas",
        "roc_auc_cv":"ROC-AUC", "tiempo_busqueda_s":"Tiempo de búsqueda (s)",
        "evaluaciones":"Configuraciones evaluadas",
    }
    st.subheader("Resultados de la búsqueda")
    st.dataframe(search[display_columns].rename(columns=display_names).round(4), hide_index=True, width="stretch")
    a, b = st.columns(2)
    with a:
        show_chart(px.bar(search, x="estrategia", y="f1_cv", color="estrategia",
                          text="f1_cv", title="F1 de validación por estrategia",
                          labels={"estrategia":"Método", "f1_cv":"F1"}))
    with b:
        show_chart(px.bar(search, x="estrategia", y="tiempo_busqueda_s", color="estrategia",
                          text="tiempo_busqueda_s", title="Tiempo de optimización",
                          labels={"estrategia":"Método", "tiempo_busqueda_s":"Segundos"}))
    st.subheader("Presupuesto de optimización")
    budget_cols = [c for c in ("estrategia", "evaluaciones", "configuraciones_completas",
                        "configuraciones_podadas", "recurso_total_arboles_por_fold",
                        "equivalentes_180_arboles") if c in search.columns]
    if budget_cols:
        st.dataframe(search[budget_cols].rename(columns={
            "estrategia":"Estrategia", "evaluaciones":"Candidatos",
            "configuraciones_completas":"Evaluaciones completas",
            "configuraciones_podadas":"Paradas anticipadas",
            "recurso_total_arboles_por_fold":"Árboles construidos por fold (total)",
            "equivalentes_180_arboles":"Equivalentes de 180 árboles"
        }).round(2), hide_index=True, width="stretch")
    st.caption("Hyperband (Optuna) incrementa el recurso de 20 a 60 y 180 árboles; puede descartar candidatos antes del presupuesto completo. Las tres estrategias proponen 16 candidatos con los mismos folds, pero no consumen el mismo número de árboles ni examinan exactamente el mismo espacio.")
    history = optional_csv("historial_busquedas.csv")
    if history is not None and "estrategia" in history:
        with st.expander("Resultados detallados por configuración"):
            relevant = [c for c in ("estrategia", "evaluacion", "f1_cv", "roc_auc_cv", "estado", "es_evaluacion_completa", "recurso_arboles", "params") if c in history.columns]
            st.dataframe(history[relevant].round(4), hide_index=True, width="stretch")
    saved_image("convergencia_busqueda.png", "Evolución del mejor F1 alcanzado durante la búsqueda")
    if {"estrategia", "f1_cv"}.issubset(search.columns):
        best = search.sort_values("f1_cv", ascending=False).iloc[0]
        insight(f"<strong>Mejor F1 en validación:</strong> {best['estrategia']} alcanzó {best['f1_cv']:.4f}. "
                "Este resultado corresponde al conjunto de desarrollo, no a una garantía de desempeño futuro.")
    with st.expander("Cómo se evaluaron las alternativas"):
        st.dataframe(pd.DataFrame([
            ("Grid Search", "Examina todas las opciones de una cuadrícula fija.", "16"),
            ("Random Search", "Prueba combinaciones seleccionadas al azar.", "16"),
            ("Hyperband (Optuna)", "Asigna más árboles a configuraciones competitivas y realiza poda anticipada.", "16"),
            ("NSGA-II", "Evalúa compromisos entre calidad y tiempo de predicción.", "16"),
        ], columns=["Método", "Enfoque", "Evaluaciones"]), hide_index=True, width="stretch")
        st.markdown("**Parámetros explorados**")
        st.dataframe(pd.DataFrame([
            ("n_estimators", "60, 180", "40, 60, 80, 100, 120, 160, 180"),
            ("max_depth", "6, 14", "4, 6, 8, 10, 14, 20, None"),
            ("min_samples_leaf", "1, 4", "1, 2, 3, 4, 6"),
            ("max_features", "sqrt, 0.75", "sqrt, 0.50, 0.75"),
        ], columns=["Hiperparámetro", "Grid Search", "Random Search / NSGA-II"]), hide_index=True, width="stretch")
        st.caption("Se utilizan los mismos tres folds estratificados del conjunto de desarrollo. Grid Search explora una cuadrícula más acotada; Hyperband puede detener ensayos anticipadamente y por ello el presupuesto computacional efectivo difiere del de Random Search.")

# 05 · PARETO
elif selected == "NSGA-II · Pareto Frontier":
    page_title(
        "EFICIENCIA OPERATIVA", "NSGA-II · Pareto Frontier",
        "Cada configuración combina una capacidad predictiva y un tiempo de respuesta. El análisis multiobjetivo identifica opciones competitivas en ambas dimensiones.",
    )
    mo = csv_artifact("nsga_todos.csv").copy()
    pareto = csv_artifact("frente_pareto.csv")
    choice = artifact_json("seleccion_pareto.json", {})
    mo["Solución"] = np.where(mo["no_dominada"].astype(str).str.lower().eq("true"),
                              "No dominada", "Dominada")
    fig = px.scatter(mo, x="latencia_ms_cliente", y="f1_cv", color="Solución",
                     hover_data=[x for x in ("trial", "roc_auc_cv") if x in mo.columns],
                     color_discrete_map={"No dominada":"#27C3B8", "Dominada":"#94A3B8"},
                     title="Frontera de eficiencia: más calidad y menos latencia",
                     labels={"latencia_ms_cliente":"Milisegundos por cliente", "f1_cv":"F1 de validación"})
    if choice and "trial" in choice and "trial" in mo.columns:
        match = mo.loc[mo["trial"].eq(choice["trial"])]
        if not match.empty:
            point = match.iloc[0]
            fig.add_trace(go.Scatter(
                x=[point["latencia_ms_cliente"]], y=[point["f1_cv"]], mode="markers",
                name="Configuración seleccionada",
                marker=dict(size=19, color="#FFCC73", symbol="star", line=dict(color="#203041", width=1)),
            ))
    show_chart(fig)
    a, b, c = st.columns(3)
    a.metric("Opciones analizadas", f"{len(mo)}")
    b.metric("Opciones eficientes", f"{len(pareto)}")
    c.metric("F1 de opción elegida", f"{choice['f1_cv']:.3f}" if "f1_cv" in choice else "—")
    insight(
        "<strong>Lectura del gráfico.</strong> Cuanto más arriba, mejor es el F1; cuanto más a la izquierda, menor es el tiempo de respuesta. "
        "Una solución no dominada no puede mejorarse en una dimensión sin perder competitividad en la otra, dentro de las alternativas evaluadas."
    )
    st.subheader("Configuraciones eficientes")
    st.dataframe(pareto[[c for c in ("trial","f1_cv","latencia_ms_cliente","roc_auc_cv") if c in pareto]].round(5).rename(
        columns={"trial":"Identificador", "f1_cv":"F1", "latencia_ms_cliente":"Latencia (ms/cliente)", "roc_auc_cv":"ROC-AUC"}
    ), hide_index=True, width="stretch")
    with st.expander("Criterio de selección y alcance de las mediciones"):
        st.markdown(
            "Se priorizó la menor latencia entre soluciones cuyo F1 se encontraba a un máximo de 0,01 puntos del mejor valor del frente. "
            "La opción se adoptó si estaba a no más de 0,015 puntos de F1 respecto al mejor bosque aleatorio encontrado por las búsquedas de un solo objetivo."
        )
        st.caption(f"Configuración en servicio: {selection.get('nombre','Bosque aleatorio optimizado')}. {selection.get('motivo','')}")
        st.caption("La latencia incluye el procesamiento completo. Puede variar con el hardware, la carga y el tamaño de los lotes.")

# 06 · EXPLICABILIDAD
elif selected == "Explainable AI · SHAP y LIME":
    page_title(
        "INTELIGENCIA EXPLICABLE", "Explainable AI · SHAP y LIME",
        "Identifica las señales que más influyen en las estimaciones y consulta ejemplos de explicación de clientes individuales.",
    )
    shap = csv_artifact("shap_global.csv")
    importance = csv_artifact("rf_feature_importance.csv")
    a, b = st.columns(2)
    with a:
        subset = shap.head(10).iloc[::-1]
        show_chart(px.bar(subset, x="SHAP_importancia_abs_media", y="Variable", orientation="h",
                          title="Importancia global · SHAP",
                          labels={"SHAP_importancia_abs_media":"Contribución media absoluta"}))
    with b:
        subset = importance.head(10).iloc[::-1]
        show_chart(px.bar(subset, x="RandomForest_importancia_impureza", y="Variable", orientation="h",
                          title="Random Forest · Feature Importance",
                          labels={"RandomForest_importancia_impureza":"Importancia relativa"}))
    insight(
        "<strong>Lectura de las variables.</strong> Los gráficos muestran importancia predictiva, no causas comprobadas de abandono. "
        "Diferentes técnicas pueden destacar variables distintas porque analizan aspectos diferentes del modelo."
    )
    st.subheader("Ejemplos de decisiones individuales")
    examples = artifact_json("explicaciones_clientes.json", [])
    for i in (1, 2):
        label = f"Perfil {i}"
        if len(examples) >= i and "probabilidad" in examples[i-1]:
            label += f" · riesgo estimado: {examples[i-1]['probabilidad']:.1%}"
        with st.expander(label, expanded=i==1):
            left, right = st.columns(2)
            shap_data = optional_csv(f"shap_cliente_{i}.csv")
            lime_data = optional_csv(f"lime_cliente_{i}.csv")
            with left:
                st.markdown("**SHAP · contribuciones a la predicción**")
                if shap_data is not None and {"feature", "SHAP_churn_1"}.issubset(shap_data.columns):
                    s = shap_data.head(12).iloc[::-1]
                    show_chart(px.bar(s, x="SHAP_churn_1", y="feature", orientation="h",
                                      color="SHAP_churn_1", color_continuous_scale="RdBu_r",
                                      title="Contribución hacia abandono"), key=f"shap_saved_{i}")
                else:
                    saved_image(f"shap_cliente_{i}.png")
            with right:
                st.markdown("**LIME · factores del entorno cercano**")
                if lime_data is not None and {"regla_lime", "peso_lime_churn_1"}.issubset(lime_data.columns):
                    s = lime_data.head(12).iloc[::-1]
                    show_chart(px.bar(s, x="peso_lime_churn_1", y="regla_lime", orientation="h",
                                      title="Influencia local sobre abandono"), key=f"lime_saved_{i}")
                else:
                    saved_image(f"lime_cliente_{i}.png")
    with st.expander("Cómo interpretar SHAP, LIME y la importancia de variables"):
        st.dataframe(pd.DataFrame([
            ("SHAP", "Cuantifica aportaciones a cada predicción y resume su magnitud global.", "Depende del escenario de referencia y de supuestos sobre variables relacionadas."),
            ("LIME", "Aproxima el modelo alrededor de un cliente concreto.", "Las muestras generadas localmente pueden afectar la explicación."),
            ("Random Forest Feature Importance", "Resume la reducción de impureza asociada a cada variable.", "No indica dirección local ni efectos causales."),
        ], columns=["Método", "Uso", "Consideración"]), hide_index=True, width="stretch")
        st.caption("Las explicaciones se interpretan para la probabilidad de abandono (clase positiva). Un efecto positivo incrementa el riesgo estimado con respecto a la referencia.")

# 07 · PREDICCIÓN
elif selected == "Prediction & Explainability":
    page_title(
        "CONSULTA INDIVIDUAL", "Prediction & Explainability",
        "Introduce un perfil de cliente para obtener su probabilidad estimada de abandono y los factores que más contribuyen al resultado.",
    )
    group_profile = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure"]
    group_services = ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                      "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    group_contract = ["Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]
    labels = {
        "gender":"Género", "SeniorCitizen":"Adulto mayor", "Partner":"Tiene pareja",
        "Dependents":"Personas dependientes", "tenure":"Antigüedad (meses)",
        "PhoneService":"Telefonía", "MultipleLines":"Múltiples líneas", "InternetService":"Internet",
        "OnlineSecurity":"Seguridad en línea", "OnlineBackup":"Respaldo en la nube",
        "DeviceProtection":"Protección de equipos", "TechSupport":"Soporte técnico",
        "StreamingTV":"Televisión por streaming", "StreamingMovies":"Películas por streaming",
        "Contract":"Contrato", "PaperlessBilling":"Facturación digital", "PaymentMethod":"Método de pago",
        "MonthlyCharges":"Cargo mensual (USD)", "TotalCharges":"Cargos acumulados (USD)",
    }
    defaults = {"Contract":"Month-to-month", "InternetService":"Fiber optic",
                "PaymentMethod":"Electronic check"}
    record = {}
    with st.form("client_form", clear_on_submit=False):
        for name, fields in [
            ("Perfil del cliente", group_profile),
            ("Servicios contratados", group_services),
            ("Contrato y facturación", group_contract),
        ]:
            st.markdown(f"#### {name}")
            a, b = st.columns(2)
            for j, field in enumerate(fields):
                holder = a if j % 2 == 0 else b
                if field in NUMERIC:
                    if field == "tenure":
                        record[field] = holder.number_input(labels[field], min_value=0,
                                                             max_value=120, value=8, step=1)
                    elif field == "MonthlyCharges":
                        record[field] = holder.number_input(labels[field], min_value=0.,
                                                             max_value=300., value=78., step=1.)
                    else:
                        record[field] = holder.number_input(labels[field], min_value=0.,
                                                             max_value=30000., value=624., step=5.)
                else:
                    levels = sorted(df[field].dropna().astype(str).unique().tolist())
                    index = levels.index(defaults[field]) if field in defaults and defaults[field] in levels else 0
                    record[field] = holder.selectbox(labels[field], levels, index=index, format_func=translate)
        submit = st.form_submit_button("Calcular riesgo de abandono", type="primary", width="stretch")

    if submit:
        from src.xai import class_1_shap, lime_for_new_customer
        customer = format_user_input(record)
        pipeline = model()
        with st.spinner("Analizando perfil..."):
            probability = float(pipeline.predict_proba(customer)[0, 1])
        a, b = st.columns(2)
        a.metric("Probabilidad estimada de abandono", f"{probability:.1%}")
        b.metric("Clasificación del modelo", "Riesgo de abandono" if probability >= .5 else "Permanencia probable")
        st.progress(float(np.clip(probability, 0., 1.)))
        st.caption("Clasificación basada en un umbral de 50 %. El riesgo es una estimación y no implica que el cliente necesariamente abandonará el servicio.")
        st.subheader("Factores que explican la estimación")
        left, right = st.columns(2)
        with left:
            st.markdown("**SHAP · contribución por variable**")
            try:
                with st.spinner("Calculando explicación SHAP..."):
                    sh = class_1_shap(pipeline, customer)
                explanation = pd.DataFrame({"Variable":sh["names"], "Contribución":sh["values"][0]})
                explanation["Magnitud"] = explanation["Contribución"].abs()
                subset = explanation.nlargest(12, "Magnitud").iloc[::-1]
                show_chart(px.bar(subset, x="Contribución", y="Variable", orientation="h",
                                  color="Contribución", color_continuous_scale="RdBu_r",
                                  title="Impacto sobre el riesgo estimado"))
            except Exception:
                LOGGER.exception("SHAP prediction failed")
                st.caption("La explicación SHAP no está disponible para esta consulta.")
        with right:
            st.markdown("**LIME · explicación local**")
            try:
                with st.spinner("Calculando explicación LIME..."):
                    explanation = lime_for_new_customer(pipeline, xai_background(), customer)
                show_chart(px.bar(explanation.iloc[::-1], x="Peso hacia abandono", y="Regla",
                                  orientation="h", title="Factores locales del perfil"))
            except Exception:
                LOGGER.exception("LIME prediction failed")
                st.caption("La explicación LIME no está disponible para esta consulta.")
        st.caption("Los factores destacados ayudan a interpretar el pronóstico; no constituyen recomendaciones automáticas ni pruebas de causalidad.")

# 08 · PLANIFICACIÓN
elif selected == "Business Application":
    page_title(
        "BUSINESS APPLICATION", "Planificar campañas de retención",
        "Simula alternativas de contacto, asigna recursos y consulta qué clientes tendrían mayor beneficio esperado bajo los supuestos indicados.",
    )
    clients = csv_artifact("riesgos_clientes_demo.csv")
    a, b = st.columns(2)
    contact_cost = a.slider("Costo por contacto (USD)", 1., 30., 8., 1.)
    effectiveness = b.slider("Eficacia incremental hipotética de campaña", 0., .50, .15, .01,
                             format="%.2f", help="Hipótesis comercial. No ha sido medida con un experimento de intervención.")
    margin = a.slider("Margen bruto estimado", .05, .80, .40, .05, format="%.2f")
    horizon = b.slider("Horizonte de análisis (meses)", 1, 18, 6)
    budget = a.slider("Presupuesto de contacto (USD)", 100., 3000., 500., 50.)
    fee = b.number_input("Cuota mensual de la plataforma (USD)", min_value=0., max_value=1000.,
                         value=59., step=10.)
    selected_clients, scenario = simulate_retention(
        clients, contact_cost, effectiveness, margin, horizon, budget, fee,
    )
    a, b, c = st.columns(3)
    a.metric("Clientes priorizados", f"{scenario['contactados']:,}")
    b.metric("Resultado neto del escenario", f"USD {scenario['resultado_neto_hipotetico_usd']:,.2f}")
    c.metric("ROI del escenario", f"{scenario['roi_hipotetico']:.1%}")
    st.caption(
        "Proyección condicional basada en parámetros editables. No representa ahorros observados ni retorno económico demostrado."
    )
    if scenario.get("eficacia_minima_break_even") is not None:
        st.write(f"Efectividad incremental estimada necesaria para cubrir costos: **{scenario['eficacia_minima_break_even']:.1%}**")
    st.subheader("Cartera priorizada")
    view = selected_clients[[c for c in (
        "customerID", "Contract", "MonthlyCharges", "probabilidad_churn", "beneficio_neto_esperado"
    ) if c in selected_clients.columns]].head(60).copy()
    view = view.rename(columns={
        "customerID":"Cliente", "Contract":"Contrato", "MonthlyCharges":"Cargo mensual (USD)",
        "probabilidad_churn":"Riesgo estimado", "beneficio_neto_esperado":"Beneficio estimado (USD)",
    })
    if "Contrato" in view.columns:
        view["Contrato"] = view["Contrato"].map(translate)
    st.dataframe(view.round(3), hide_index=True, width="stretch")
    download_table(selected_clients, "retenia_escenario_comercial.csv", "Exportar cartera priorizada")
    with st.expander("Interpretación y alcance del escenario"):
        st.markdown(
            "El beneficio bruto por cliente se aproxima con la probabilidad de abandono, la efectividad incremental supuesta, "
            "el cargo mensual, el margen y el horizonte. A continuación se descuentan los costos de contacto y plataforma."
        )
        st.latex(r"B_i = p_i\,e\,M_i\,m\,h - C_{\mathrm{contacto}}")
        st.write("La efectividad real de una campaña necesita validarse con un piloto controlado. Las decisiones de contacto requieren además políticas de privacidad, consentimiento y revisión comercial.")

# 09 · INFORMACIÓN DEL SERVICIO
else:
    page_title(
        "TRANSPARENCIA Y GOBERNANZA", "Información del servicio",
        "Alcance de RetenIA, evaluación de calidad y criterios para su utilización responsable en una operación comercial.",
    )
    a, b = st.columns(2)
    with a.container(border=True):
        st.markdown("#### Nuestra propuesta")
        st.write(
            "Una solución de inteligencia comercial para departamentos de retención y CRM en telecomunicaciones. "
            "Integra análisis de clientes, estimación de abandono, interpretabilidad y simulación de campañas."
        )
        st.markdown("**Usuarios previstos:** líderes de CRM, analistas comerciales y responsables de retención.")
        st.markdown("**Modalidad comercial propuesta:** suscripción y piloto de validación.")
    with b.container(border=True):
        st.markdown("#### Uso responsable")
        st.write(
            "Las predicciones no equivalen a certezas ni demuestran causas. "
            "La priorización debe ser supervisada por responsables de negocio y evaluada con indicadores reales."
        )
        st.markdown("**Datos:** demostración con un conjunto histórico de telecomunicaciones.")
        st.markdown("**Valor económico:** escenarios sujetos a hipótesis explícitas.")
    st.subheader("Detalles del modelo")
    method = selection.get("nombre", "Bosque aleatorio optimizado")
    st.write(f"**Tecnología utilizada:** {method}.")
    st.write("**Objetivo del modelo:** estimar la probabilidad de abandono de un cliente (Churn = 1).")
    st.write("**Tratamiento:** preparación numérica y categórica integrada en el pipeline de predicción.")
    st.write("**Validación:** validación cruzada estratificada y evaluación independiente posterior.")
    if metrics.get("modelo_final"):
        short_metrics(metrics["modelo_final"])
    st.subheader("Consideraciones para la adopción")
    st.markdown(
        "**Representatividad:** verificar que los clientes actuales se parezcan a la población histórica empleada.\n\n"
        "**Equidad y privacidad:** evaluar sesgos, permisos de tratamiento y protección de la información.\n\n"
        "**Seguimiento:** controlar cambios en la distribución de variables y en la calidad de las predicciones.\n\n"
        "**Efectividad comercial:** medir resultados incrementales mediante una prueba controlada antes de escalar la campaña."
    )
    st.caption("Fuente de datos: Telco Customer Churn (Kaggle/IBM). Las cifras de rendimiento corresponden a una evaluación experimental del conjunto histórico.")

st.divider()
st.caption("RetenIA · Inteligencia para retención de clientes  |  Estimaciones sujetas a incertidumbre y revisión humana.")
