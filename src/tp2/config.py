"""Constantes globales: rutas, semillas y esquema de validación."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw" / "bank-additional-full.csv"
FIGURES = ROOT / "reports" / "figures"
MODELS = ROOT / "models"
CACHE = ROOT / ".cache"  # resultados de CV cacheados (no versionar)

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
N_JOBS = -1  # todos los cores

TARGET = "y"

# Leakage: `duration` sólo se conoce después de la llamada -> fuera del modelo realista.
LEAKAGE_COLS = ["duration"]

NUMERIC_COLS = [
    "age", "campaign", "pdays", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
]
CATEGORICAL_COLS = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]

# Grupos de columnas surgidos del EDA (para probar con / sin cada uno)
FLAT_COLS = ["housing", "loan", "day_of_week"]  # tasa de yes ~igual en todas sus categorías
MACRO_REDUNDANT = ["emp.var.rate", "euribor3m"]  # rho > 0.9 con nr.employed

# Métricas (clase 4). Con ~11 % de positivos, accuracy no sirve.
#   ROC-AUC: no depende del umbral -> elegir modelo.
#   F1: balance precision/recall con el umbral elegido.
METRICS = ["roc_auc", "f1"]
