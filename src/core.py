"""Limpieza y preprocesamiento alineados con el EDA original de Telco Customer Churn.

SEED=42; split 80/20 estratificado; TotalCharges faltante con tenure=0 => 0;
SeniorCitizen semántico; 3 numéricas, 16 categóricas; sin borrar filas.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

SEED = 42
TARGET = "Churn"
ID = "customerID"
CSV_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod"
]
FEATURES = NUMERIC + CATEGORICAL
EXPECTED = set(FEATURES + [TARGET, ID])
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def locate_csv(download_if_missing: bool = True) -> Path:
    candidates = [PROJECT_ROOT / 'data' / CSV_NAME, Path.cwd() / 'data' / CSV_NAME,
                  Path.cwd() / CSV_NAME, PROJECT_ROOT / CSV_NAME]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if download_if_missing:
        try:
            import kagglehub
        except ImportError as exc:
            raise RuntimeError('No está disponible kagglehub y no se encontró el CSV. Instala requirements.txt o guarda el CSV en data/.') from exc
        download_path = Path(kagglehub.dataset_download('blastchar/telco-customer-churn'))
        matches = list(download_path.rglob(CSV_NAME))
        if len(matches) == 1:
            return matches[0]
    raise FileNotFoundError('Descarga el dataset blastchar/telco-customer-churn de Kaggle y guarda su CSV en data/. No se usarán datos sintéticos como resultados.')


def load_telco(csv_path: Path | str | None = None, download_if_missing: bool = True):
    path = Path(csv_path) if csv_path is not None else locate_csv(download_if_missing)
    raw_bytes = path.read_bytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    df = pd.read_csv(path, dtype={'customerID': 'string'})
    if set(df.columns) != EXPECTED:
        raise ValueError(f'Columnas inesperadas; faltantes={EXPECTED-set(df.columns)}; extra={set(df.columns)-EXPECTED}')
    if df[ID].duplicated().any() or df.duplicated().any():
        raise ValueError('El CSV contiene IDs o filas duplicadas: revisar antes de continuar')
    for col in df.select_dtypes(include=['object','string']).columns:
        df[col] = df[col].astype('string').str.strip().replace('', pd.NA)
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    missing_total = df['TotalCharges'].isna()
    valid_zero = missing_total & df['tenure'].eq(0)
    df.loc[valid_zero, 'TotalCharges'] = 0.0
    if (missing_total & ~df['tenure'].eq(0)).any():
        raise ValueError('Hay TotalCharges nulos con tenure != 0: no imputar arbitrariamente')
    codes = df['SeniorCitizen'].astype('string')
    if not set(codes.dropna().unique()).issubset({'0','1'}):
        raise ValueError('Códigos anómalos en SeniorCitizen')
    df['SeniorCitizen'] = codes.map({'0': 'No', '1': 'Yes'}).astype('string')
    if df[TARGET].isna().any() or set(df[TARGET].unique()) != {'No','Yes'}:
        raise ValueError('Churn tiene etiquetas inesperadas o vacías')
    df[TARGET] = df[TARGET].map({'No':0,'Yes':1}).astype(int)
    if df[FEATURES].isna().any().any():
        raise ValueError(f'Valores pendientes no contemplados en el EDA: {df[FEATURES].isna().sum().to_dict()}')
    # Controles del EDA: la fuente original no requiere eliminar registros.
    if (df[NUMERIC] < 0).any().any():
        raise ValueError('Importes/antigüedad negativos; revisar fuente')
    audit = {
        'csv': str(path), 'sha256': sha256, 'rows': int(len(df)),
        'columns_original': int(df.shape[1]), 'features': int(len(FEATURES)),
        'numeric_count': len(NUMERIC), 'categorical_count': len(CATEGORICAL),
        'churn_yes': int(df[TARGET].sum()), 'churn_no': int((1-df[TARGET]).sum()),
        'churn_rate': float(df[TARGET].mean()), 'totalcharges_filled_zero': int(valid_zero.sum()),
        'rows_dropped': 0, 'features_dropped': [ID],
        'note': 'Correcciones determinísticas y basadas en dominio; imputación/escala/OHE dentro de CV.'
    }
    return df, audit


def split_data(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=.20, stratify=df[TARGET], random_state=SEED)
    assert set(X_train.index).isdisjoint(X_test.index)
    return X_train, X_test, y_train, y_test


def make_preprocessor():
    numerical = Pipeline([
        ('imputador', SimpleImputer(strategy='median')),
        ('escalador', StandardScaler())
    ])
    categorical = Pipeline([
        ('imputador', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    return ColumnTransformer(
        [('num',numerical,NUMERIC),('cat',categorical,CATEGORICAL)],
        remainder='drop', verbose_feature_names_out=True)


def cv_splitter():
    return StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)


def format_user_input(record: dict) -> pd.DataFrame:
    """Transforma el formulario a un registro con las 19 variables originales y validadas."""
    frame = pd.DataFrame([{name: record[name] for name in FEATURES}])
    frame['SeniorCitizen'] = frame['SeniorCitizen'].astype(str)
    for col in NUMERIC:
        frame[col] = pd.to_numeric(frame[col], errors='raise')
    return frame
