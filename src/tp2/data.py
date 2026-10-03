"""Carga del dataset y split train/test (antes de cualquier transformación)."""

import pandas as pd
from sklearn.model_selection import train_test_split

from tp2.config import CATEGORICAL_COLS, DATA_RAW, RANDOM_STATE, TARGET, TEST_SIZE


def load_raw(path=DATA_RAW, drop_duplicates: bool = True) -> pd.DataFrame:
    """Lee el CSV (sep=';'), tipa categóricas y codifica y como 0/1."""
    df = pd.read_csv(path, sep=";")
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].astype("category")
    df[TARGET] = (df[TARGET] == "yes").astype("int8")
    return df


def split_xy(df: pd.DataFrame):
    return df.drop(columns=TARGET), df[TARGET]


def train_test(df: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """Split estratificado. El test queda guardado hasta el modelo final (punto 4)."""
    X, y = split_xy(df)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def stratified_subsample(X, y, n: int, random_state: int = RANDOM_STATE):
    """Submuestra estratificada para modelos O(n²) (SVC-RBF) en curvas de validación."""
    if n >= len(y):
        return X, y
    Xs, _, ys, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=random_state)
    return Xs, ys
