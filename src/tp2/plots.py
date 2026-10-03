"""Gráficos reutilizables (la consigna pide gráficos con valores, no tablas)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, roc_curve

from tp2.config import FIGURES

# Paleta fija (validada para daltonismo): identidad de clase / serie, nunca por ranking.
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8f98"
CLASS_COLORS = {0: BLUE, 1: ORANGE}  # no / yes
SERIES = [BLUE, ORANGE, "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def setup():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.bbox": "tight", "axes.prop_cycle": plt.cycler(color=SERIES),
        "grid.color": "#e6e6e6", "axes.edgecolor": "#cccccc",
    })


def save(fig, name: str):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png")


# ---------------------------------------------------------------- EDA

def class_balance(y: pd.Series, ax=None):
    ax = ax or plt.gca()
    vc = y.value_counts().sort_index()
    pct = vc / vc.sum() * 100
    bars = ax.bar(["no (0)", "yes (1)"], pct.values, color=[BLUE, ORANGE], width=0.6)
    ax.bar_label(bars, labels=[f"{p:.1f}%\n(n={n:,})" for p, n in zip(pct, vc, strict=True)])
    ax.set_ylabel("% de clientes")
    ax.set_ylim(0, 100)
    ax.set_title("Balance de clases (train)")
    return ax


def unknown_share(df: pd.DataFrame, cols, token="unknown", ax=None):
    """% de 'unknown' por columna categórica (los faltantes vienen codificados así)."""
    ax = ax or plt.gca()
    s = (df[cols] == token).mean().mul(100).sort_values()
    s = s[s > 0]
    bars = ax.barh(s.index, s.values, color=BLUE)
    ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=9)
    ax.set_xlabel("% 'unknown'")
    ax.set_title("Valores faltantes codificados como 'unknown'")
    return ax


def target_rate_by(df: pd.DataFrame, col: str, target="y", ax=None, min_n: int = 0):
    """Tasa de 'yes' por categoría (con n), útil para decidir agrupaciones/exclusiones."""
    ax = ax or plt.gca()
    g = df.groupby(col, observed=True)[target].agg(["mean", "size"])
    g = g[g["size"] >= min_n].sort_values("mean")
    bars = ax.barh(g.index.astype(str), g["mean"] * 100, color=BLUE)
    labels = [f"{m * 100:.1f}% (n={n})" for m, n in zip(g["mean"], g["size"], strict=True)]
    ax.bar_label(bars, labels=labels, fontsize=8, padding=2)
    ax.axvline(df[target].mean() * 100, ls="--", c=GRAY, lw=1)
    ax.set_xlim(0, max(g["mean"].max() * 100 * 1.45, 1))
    ax.set_xlabel("% yes  (línea = tasa global)")
    ax.set_title(col)
    return ax


def target_rate_grid(df: pd.DataFrame, cols, target="y", ncols=3):
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.2 * nrows))
    for ax, c in zip(axes.flat, cols, strict=False):
        target_rate_by(df, c, target, ax=ax)
    for ax in axes.flat[len(cols):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def numeric_by_class(df: pd.DataFrame, cols, target="y", ncols=3, bins=40):
    """Histogramas normalizados por clase: forma, escala, asimetría y separación."""
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3 * nrows))
    for ax, c in zip(axes.flat, cols, strict=False):
        for k, lab in [(0, "no"), (1, "yes")]:
            ax.hist(df.loc[df[target] == k, c], bins=bins, density=True, histtype="step",
                    lw=2, color=CLASS_COLORS[k], label=lab)
        ax.set_title(f"{c}  (skew={df[c].skew():.1f})")
        ax.set_yticks([])
    axes.flat[0].legend(title="y")
    for ax in axes.flat[len(cols):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def corr_heatmap(df: pd.DataFrame, cols, method="pearson", ax=None):
    ax = ax or plt.gca()
    c = df[cols].corr(method=method)
    mask = np.triu(np.ones_like(c, dtype=bool), k=1)
    sns.heatmap(c, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                center=0, square=True, cbar_kws={"shrink": 0.7}, annot_kws={"size": 8}, ax=ax)
    ax.set_title(f"Correlación ({method})")
    return ax


def variant_comparison(res: pd.DataFrame, metric="roc_auc", base="base"):
    """Barras agrupadas (media ± std entre folds): métrica CV por modelo y variante."""
    d = res[res.metric == metric]
    mean = d.pivot(index="model", columns="variant", values="mean")
    std = d.pivot(index="model", columns="variant", values="std")
    cols = [base] + [c for c in mean.columns if c != base]
    fig, ax = plt.subplots(figsize=(2.4 * len(cols) + 3, 4))
    w = 0.8 / len(cols)
    x = np.arange(len(mean))
    for i, v in enumerate(cols):
        bars = ax.bar(x + i * w, mean[v], w * 0.92, yerr=std[v], capsize=3, label=v,
                      color=SERIES[i % len(SERIES)], error_kw={"lw": 1})
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=1)
    ax.set_xticks(x + w * (len(cols) - 1) / 2, mean.index)
    ax.set_ylabel(f"{metric} (CV, train)")
    ax.set_ylim(0, (mean + std).max().max() * 1.15)
    ax.legend(title="variante", ncols=len(cols), loc="upper center", bbox_to_anchor=(0.5, -0.1),
              frameon=False)
    ax.set_title(f"Efecto de las decisiones de preprocesamiento — {metric} (media ± std)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- Modelos

MODEL_COLORS = {"NaiveBayes": BLUE, "SVM": ORANGE, "KNN": "#1baf7a", "RF": "#eda100"}
MODEL_ORDER = ["NaiveBayes", "SVM", "KNN", "RF"]


def cv_bars(folds: pd.DataFrame, metrics=("roc_auc", "f1"), baseline=None,
            titles=None):
    """Media (barra) ± std (error) y cada fold (puntos, unidos por modelo = mismos folds).
    baseline: {metric: valor} -> línea horizontal de referencia (p. ej. azar)."""
    baseline = baseline or {}
    titles = titles or {}
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4), squeeze=False)
    for ax, m in zip(axes[0], metrics, strict=True):
        d = folds[folds.metric == m]
        g = d.groupby("model")["score"].agg(["mean", "std"]).reindex(MODEL_ORDER).dropna()
        x = np.arange(len(g))
        bars = ax.bar(x, g["mean"], yerr=g["std"], capsize=4, width=0.6,
                      color=[MODEL_COLORS[k] for k in g.index], error_kw={"lw": 1.2})
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in g["mean"]], padding=3, fontsize=9,
                     fontweight="bold")
        for i, k in enumerate(g.index):
            pts = d[d.model == k]["score"].to_numpy()
            ax.scatter(np.full(len(pts), i) + np.linspace(-0.12, 0.12, len(pts)), pts,
                       s=12, c="k", zorder=3)
        if m in baseline:
            ax.axhline(baseline[m], ls="--", c=GRAY, lw=1.2, label=f"azar = {baseline[m]:.2f}")
            ax.legend(loc="upper left", fontsize=8)
        ax.set_xticks(x, g.index)
        ax.set_ylim(0, max(1e-9, (g["mean"] + g["std"]).max()) * 1.2)
        ax.set_title(titles.get(m, f"{m} (CV 5-fold, train)"))
    fig.tight_layout()
    return fig


def roc_curves_oof(oof: dict, y, thresholds: pd.DataFrame | None = None, ax=None):
    """Curvas ROC con las predicciones de validación de la CV; punto = umbral elegido."""
    ax = ax or plt.gca()
    for k in [m for m in MODEL_ORDER if m in oof]:
        fpr, tpr, _ = roc_curve(y, oof[k])
        auc = roc_auc_score(y, oof[k])
        ax.plot(fpr, tpr, lw=2, c=MODEL_COLORS[k], label=f"{k} (AUC={auc:.3f})")
        if thresholds is not None:
            t = thresholds.loc[k]
            ax.plot(t["fpr"], t["tpr"], "o", ms=8, c=MODEL_COLORS[k], mec="white", mew=1.5)
    ax.plot([0, 1], [0, 1], ls="--", c=GRAY, lw=1.2, label="azar (AUC=0.5)")
    ax.set(xlabel="FPR", ylabel="TPR (recall)", xlim=(-0.01, 1), ylim=(0, 1.01),
           title="Curvas ROC (CV, train) · punto = umbral elegido")
    ax.legend(fontsize=8, loc="lower right")
    return ax


def default_vs_tuned(folds: pd.DataFrame, metric="f1", ax=None):
    """Efecto del umbral: métrica con umbral por defecto vs. umbral elegido en train."""
    ax = ax or plt.gca()
    g = (folds[folds.metric.isin([f"{metric}@default", metric])]
         .groupby(["model", "metric"])["score"].mean().unstack().reindex(MODEL_ORDER).dropna())
    x = np.arange(len(g))
    for i, (col, c, lab) in enumerate([(f"{metric}@default", GRAY, "umbral por defecto"),
                                       (metric, BLUE, "umbral elegido (F1 máx.)")]):
        bars = ax.bar(x + (i - 0.5) * 0.38, g[col], 0.36, color=c, label=lab)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x, g.index)
    ax.set_ylabel(metric)
    ax.set_ylim(0, g.values.max() * 1.25)
    ax.legend(fontsize=8)
    ax.set_title(f"{metric}: umbral por defecto vs. elegido")
    return ax


def train_vs_val(folds: pd.DataFrame, ax=None):
    """Gap de generalización (clase 7): ROC-AUC en train vs. validación."""
    ax = ax or plt.gca()
    g = (folds[folds.metric.isin(["train_roc_auc", "roc_auc"])]
         .groupby(["model", "metric"])["score"].mean().unstack().reindex(MODEL_ORDER).dropna())
    x = np.arange(len(g))
    for i, (col, c, lab) in enumerate([("train_roc_auc", ORANGE, "train"),
                                       ("roc_auc", BLUE, "validación")]):
        bars = ax.bar(x + (i - 0.5) * 0.38, g[col], 0.36, color=c, label=lab)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x, g.index)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.5, 1.02)
    ax.legend(fontsize=8)
    ax.set_title("Gap de generalización: ROC-AUC train vs. validación")
    return ax


def validation_curve(vc: pd.DataFrame, param_label: str, metric="roc_auc", ax=None):
    ax = ax or plt.gca()
    x = range(len(vc))
    for kind, c in [("train", ORANGE), ("val", BLUE)]:
        m, s = vc[f"{kind}_mean"], vc[f"{kind}_std"]
        ax.plot(x, m, "o-", c=c, lw=2, ms=6, label=kind)
        ax.fill_between(x, m - s, m + s, color=c, alpha=0.2)
    best = vc["val_mean"].idxmax()
    ax.annotate(f"{vc.val_mean[best]:.3f}", (best, vc.val_mean[best]),
                textcoords="offset points", xytext=(0, 8), ha="center")
    ax.set_xticks(list(x), vc["param"])
    ax.set_xlabel(param_label)
    ax.set_ylabel(metric)
    ax.legend()
    return ax


def final_curves(model, X_test, y_test, threshold=None):
    """Test (una sola vez): curva ROC + matriz de confusión con el umbral elegido en train."""
    from tp2.evaluation import scores_of

    s = scores_of(model, X_test)
    pred = model.predict(X_test) if threshold is None else (s >= threshold).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fpr, tpr, _ = roc_curve(y_test, s)
    axes[0].plot(fpr, tpr, lw=2, c=BLUE, label=f"AUC={roc_auc_score(y_test, s):.3f}")
    axes[0].plot([0, 1], [0, 1], ls="--", c=GRAY, lw=1.2, label="azar")
    axes[0].set(xlabel="FPR", ylabel="TPR (recall)", title="ROC en test")
    axes[0].legend()
    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=axes[1], colorbar=False,
                                            cmap="Blues", display_labels=["no", "yes"])
    axes[1].set_title("Matriz de confusión (test)")
    fig.tight_layout()
    return fig
