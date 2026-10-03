"""Los 4 clasificadores pedidos, cada uno con el preprocesamiento que le corresponde."""

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from tp2.config import N_JOBS, RANDOM_STATE
from tp2.features import make_pipeline


class MixedNB(ClassifierMixin, BaseEstimator):
    """Naive Bayes como en la clase 6:
    - atributos continuos: P(x_i | y) gaussiana (media y varianza por clase);
    - atributos categóricos: P(x_i | y) por frecuencia relativa con corrección de Laplace (alpha=1).
    Por independencia condicional: log P(y|x) ∝ log P(y) + Σ log P(x_i|y) (sumando ambas partes).
    No necesita escalado ni one-hot.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def _split(self, X):
        Xn = X[self.num_cols_].to_numpy(dtype=float)
        Xc = np.column_stack([X[c].cat.codes.to_numpy() for c in self.cat_cols_])
        return Xn, Xc

    def fit(self, X, y):
        self.cat_cols_ = [c for c in X.columns if str(X[c].dtype) == "category"]
        self.num_cols_ = [c for c in X.columns if c not in self.cat_cols_]
        Xn, Xc = self._split(X)
        self.gnb_ = GaussianNB().fit(Xn, y)
        n_cats = [len(X[c].cat.categories) for c in self.cat_cols_]  # todas las categorías posibles
        self.cnb_ = CategoricalNB(alpha=self.alpha, min_categories=n_cats).fit(Xc, y)
        self.classes_ = self.gnb_.classes_
        return self

    def predict_log_proba(self, X):
        Xn, Xc = self._split(X)
        # cada parte ya incluye log P(y): se resta una vez para no contarlo doble
        jll = (self.gnb_.predict_joint_log_proba(Xn) + self.cnb_.predict_joint_log_proba(Xc)
               - np.log(self.gnb_.class_prior_))
        return jll - logsumexp(jll, axis=1, keepdims=True)

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]


def get_models(only=None, **fe_kwargs) -> dict:
    """
    - NB: MixedNB sobre las variables crudas (gaussiana + categóricas con Laplace).
    - SVM y KNN: z-score + one-hot (dependen de distancias / productos internos).
    - RF: one-hot sin escalar (los árboles no necesitan normalización).
    - El desbalance (11 % yes) se maneja eligiendo el umbral (punto 2.3), igual para los 4.

    Rendimiento: SVC-RBF escala ~O(n²) (cache_size grande ayuda; para las curvas de validación
    se usa una submuestra). KNN y RF paralelizan con n_jobs.
    fe_kwargs (drop, keep_leakage) se pasan al FeatureEngineer. only: construir sólo algunos.
    """
    fe = fe_kwargs
    models = {
        "NaiveBayes": make_pipeline(MixedNB(alpha=1.0), kind=None, **fe),
        "SVM": make_pipeline(
            SVC(kernel="rbf", C=1.0, gamma="scale", cache_size=2000, random_state=RANDOM_STATE),
            kind="scaled", **fe,
        ),
        "KNN": make_pipeline(
            KNeighborsClassifier(n_neighbors=15, weights="uniform", n_jobs=N_JOBS),
            kind="scaled", **fe,
        ),
        "RF": make_pipeline(
            RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=N_JOBS,
                                   random_state=RANDOM_STATE),
            kind="tree", **fe,
        ),
    }
    return {k: v for k, v in models.items() if only is None or k in only}


# Grillas para curvas de validación (punto 3). Nombre del parámetro dentro del Pipeline.
PARAM_RANGES = {
    "SVM": ("model__C", [0.01, 0.1, 1, 10, 100]),
    "KNN": ("model__n_neighbors", [1, 3, 5, 11, 21, 41, 81, 151]),
    "RF": ("model__max_depth", [2, 4, 6, 8, 12, 16, 24, None]),
}
