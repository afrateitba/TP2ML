"""CV, curvas de validación y evaluación final. Resultados cacheados en disco (.cache/)."""

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed
from sklearn import metrics
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, validation_curve

from tp2.config import CACHE, N_JOBS, N_SPLITS, RANDOM_STATE

# Cachea por (pipeline, datos, parámetros): re-ejecutar el notebook no re-entrena.
memory = Memory(CACHE, verbose=0)


def _src_version() -> str:
    """Hash del código de src/tp2 (salvo plots). Se pasa a las funciones cacheadas para que
    un cambio en features.py/models.py/config.py invalide la caché automáticamente."""
    h = hashlib.md5()
    for f in sorted(Path(__file__).parent.glob("*.py")):
        if f.name != "plots.py":
            h.update(f.read_bytes())
    return h.hexdigest()[:12]


SRC_VERSION = _src_version()


def get_cv(n_splits: int = N_SPLITS) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


# ------------------------------------------------------------------ CV k-fold
def _fit_fold(model, X, y, tr, va):
    m = clone(model)
    t0 = time.perf_counter()
    m.fit(X.iloc[tr], y.iloc[tr])
    fit_time = time.perf_counter() - t0
    return {
        "va": va,
        "score": scores_of(m, X.iloc[va]),       # score continuo (proba / decision_function)
        "pred": m.predict(X.iloc[va]),           # predicción con el umbral por defecto
        "train_auc": metrics.roc_auc_score(y.iloc[tr], scores_of(m, X.iloc[tr])),
        "fit_time": fit_time,
    }


@memory.cache
def _cv_oof(model, X, y, n_splits, _v=None):
    folds = list(get_cv(n_splits).split(X, y))
    return Parallel(n_jobs=N_JOBS)(delayed(_fit_fold)(model, X, y, tr, va) for tr, va in folds)


def choose_threshold(y, s) -> dict:
    """Umbral que maximiza F1 sobre los scores de validación de la CV (train; el test no
    participa). Devuelve también su punto (FPR, TPR) para marcarlo en la curva ROC."""
    p, r, t = metrics.precision_recall_curve(y, s)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    i = int(np.nanargmax(f1[:-1]))  # el último punto no tiene umbral asociado
    pred = (s >= t[i]).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y, pred).ravel()
    return {"threshold": float(t[i]), "f1": f1[i], "precision": p[i], "recall": r[i],
            "fpr": fp / (fp + tn), "tpr": tp / (tp + fn)}


def cv_oof(models: dict, X, y, n_splits=N_SPLITS) -> dict:
    """
    K-fold estratificado sobre TRAIN, en UNA pasada por modelo. Devuelve:
      - folds: métricas por fold (ROC-AUC; F1/recall/precision con el umbral elegido y con el
        umbral por defecto; ROC-AUC en train para el gap de generalización; tiempo de fit);
      - oof: score de validación de cada fila de train (predicha por el modelo del fold que
        NO la usó para entrenar) -> curva ROC y elección del umbral;
      - thresholds: umbral elegido por modelo (el que maximiza F1).
    Todos los modelos usan los MISMOS folds -> se pueden comparar fold a fold.
    """
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)
    rows, oof, thr = [], {}, {}
    for name, model in models.items():
        folds = _cv_oof(model, X, y, n_splits, _v=SRC_VERSION)
        s_oof = np.empty(len(y))
        for f in folds:
            s_oof[f["va"]] = f["score"]
        oof[name] = s_oof
        thr[name] = choose_threshold(y, s_oof)
        t = thr[name]["threshold"]
        for k, f in enumerate(folds):
            yv, sv = y.iloc[f["va"]], f["score"]
            pred = (sv >= t).astype(int)
            vals = {
                "roc_auc": metrics.roc_auc_score(yv, sv),
                "f1": metrics.f1_score(yv, pred),
                "recall": metrics.recall_score(yv, pred),
                "precision": metrics.precision_score(yv, pred, zero_division=0),
                "%pred_yes": pred.mean() * 100,
                "f1@default": metrics.f1_score(yv, f["pred"]),
                "recall@default": metrics.recall_score(yv, f["pred"]),
                "precision@default": metrics.precision_score(yv, f["pred"], zero_division=0),
                "%pred_yes@default": f["pred"].mean() * 100,
                "train_roc_auc": f["train_auc"],
                "fit_time": f["fit_time"],
            }
            rows += [{"model": name, "fold": k, "metric": m, "score": v} for m, v in vals.items()]
    return {"folds": pd.DataFrame(rows), "oof": oof, "thresholds": pd.DataFrame(thr).T, "y": y}


def summarize_oof(folds: pd.DataFrame, metrics_=None) -> pd.DataFrame:
    """Tabla 'media ± std' por modelo (filas) y métrica (columnas)."""
    d = folds if metrics_ is None else folds[folds.metric.isin(metrics_)]
    g = d.groupby(["model", "metric"])["score"].agg(["mean", "std"])
    txt = g["mean"].map("{:.3f}".format) + " ± " + g["std"].map("{:.3f}".format)
    out = txt.unstack("metric")
    return out[metrics_] if metrics_ is not None else out


def compare_variants(variants: dict, make_models, X, y, metrics_=("roc_auc", "f1"),
                     n_splits=N_SPLITS) -> pd.DataFrame:
    """CV de cada variante de preprocesamiento. variants: {nombre: fe_kwargs};
    make_models: fe_kwargs -> dict de pipelines. Devuelve media y std por modelo/variante."""
    out = []
    for name, kw in variants.items():
        f = cv_oof(make_models(**kw), X, y, n_splits=n_splits)["folds"]
        g = f[f.metric.isin(metrics_)].groupby(["model", "metric"])["score"].agg(["mean", "std"])
        out.append(g.reset_index().assign(variant=name))
    return pd.concat(out, ignore_index=True)


@memory.cache
def _validation_curve(model, X, y, param_name, param_range, scoring, n_splits, _v=None):
    return validation_curve(
        model, X, y, param_name=param_name, param_range=param_range,
        cv=get_cv(n_splits), scoring=scoring, n_jobs=N_JOBS,
    )


def val_curve(model, X, y, param_name, param_range, scoring="roc_auc",
              n_splits=N_SPLITS) -> pd.DataFrame:
    tr, va = _validation_curve(model, X, y, param_name, list(param_range), scoring, n_splits,
                              _v=SRC_VERSION)
    return pd.DataFrame({
        "param": [str(p) for p in param_range],
        "train_mean": tr.mean(1), "train_std": tr.std(1),
        "val_mean": va.mean(1), "val_std": va.std(1),
    })


def scores_of(model, X) -> np.ndarray:
    """Score continuo (proba o decision_function) para PR/ROC."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:  # SVC con probability=False
            pass
    return model.decision_function(X)


def test_report(model, X_test, y_test, threshold: float | None = None) -> dict:
    """Usar UNA sola vez, con el modelo final ya elegido por CV.
    threshold: umbral elegido en train (CV); None = umbral por defecto de predict()."""
    s = scores_of(model, X_test)
    pred = model.predict(X_test) if threshold is None else (s >= threshold).astype(int)
    return {
        "roc_auc": metrics.roc_auc_score(y_test, s),
        "f1": metrics.f1_score(y_test, pred),
        "recall": metrics.recall_score(y_test, pred),
        "precision": metrics.precision_score(y_test, pred),
        "confusion_matrix": metrics.confusion_matrix(y_test, pred),
    }
