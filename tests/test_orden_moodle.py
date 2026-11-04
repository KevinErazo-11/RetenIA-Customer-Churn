from pathlib import Path
import ast
ROOT = Path(__file__).resolve().parents[1]

def test_portal_profesional_sin_reentrenamiento():
    source = (ROOT/'app.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    for term in ('Grid Search','Random Search','Hyperband','NSGA-II','SHAP','LIME'):
        assert term in source, term
    for term in ('guion de defensa', 'qué mostrar al profesor', 'Moodle', 'TPE (Optuna)'):
        assert term.lower() not in source.lower(), term
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr in ('fit','fit_transform','fit_resample') for n in ast.walk(tree))

def test_fuente_beamer_generada():
    source = (ROOT/'build_pitch.py').read_text(encoding='utf-8')
    assert 'documentclass[aspectratio=169' in source
    assert 'Hyperband' in source
