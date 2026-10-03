"""Feature engineering + preprocesamiento. Todo vive dentro del Pipeline -> sin leakage en CV."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tp2.config import LEAKAGE_COLS


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Transformaciones fila a fila (no aprenden nada de los datos -> no hay leakage).

    - Elimina columnas con leakage (`duration`) salvo `keep_leakage=True` (sólo benchmark).
    - Elimina las columnas de `drop` (decididas en el EDA).
    - `pdays == 999` significa "nunca contactado": nueva variable categórica `prev_contacted`
      (sí/no) y el 999 pasa a -1 para que no domine el escalado ni las distancias.
    """

    def __init__(self, drop=None, pdays_flag: bool = True, keep_leakage: bool = False):
        self.drop = drop
        self.pdays_flag = pdays_flag
        self.keep_leakage = keep_leakage

    def fit(self, X, y=None):
        self.feature_names_out_ = self.transform(X.head(1)).columns.to_numpy()
        return self

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        to_drop = list(self.drop or []) + ([] if self.keep_leakage else LEAKAGE_COLS)
        X = X.drop(columns=[c for c in to_drop if c in X.columns])
        if self.pdays_flag and "pdays" in X.columns:
            never = X["pdays"].eq(999)
            X["prev_contacted"] = pd.Categorical(np.where(never, "no", "yes"),
                                                 categories=["no", "yes"])
            X["pdays"] = np.where(never, -1, X["pdays"])
        return X


_num = make_column_selector(dtype_include="number")
_cat = make_column_selector(dtype_include=["category", "object"])


def make_preprocessor(kind: str = "scaled", min_frequency: float | int | None = 0.01):
    """
    kind="scaled": z-score + one-hot -> SVM y KNN (usan distancias/productos internos:
                   sin escalar, las variables grandes dominan).
    kind="tree":   numéricas sin escalar + one-hot -> RF (los árboles no necesitan normalización).
    min_frequency agrupa categorías raras (<1 % de train) en 'infrequent'.
    """
    cat = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=min_frequency,
        sparse_output=False,
    )
    if kind == "scaled":
        num = StandardScaler()
    elif kind == "tree":
        num = "passthrough"
    else:
        raise ValueError(f"kind desconocido: {kind}")

    return ColumnTransformer(
        [("num", num, _num), ("cat", cat, _cat)],
        verbose_feature_names_out=False,
    )


def make_pipeline(model, kind: str = "scaled", min_frequency=0.01, **fe_kwargs) -> Pipeline:
    """kind=None: sin preprocesador (Naive Bayes trabaja directo sobre numéricas y categóricas).
    fe_kwargs -> FeatureEngineer (drop, keep_leakage, pdays_flag)."""
    steps = [("fe", FeatureEngineer(**fe_kwargs))]
    if kind is not None:
        steps.append(("prep", make_preprocessor(kind, min_frequency)))
    steps.append(("model", model))
    return Pipeline(steps)
