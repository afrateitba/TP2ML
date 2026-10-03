# TP2 — Clasificación supervisada (Bank Marketing)

72.75 Aprendizaje Automático · ITBA · Defensa 07/10/26. Consigna en `docs/consigna_tp2.pdf`.

## Setup

```bash
uv sync                 # crea .venv con dependencias + grupo dev (jupyter, ruff)
uv run jupyter lab      # o en VS Code: seleccionar el kernel de .venv
```

## Estructura

```
data/raw/            dataset original (sep=';')
docs/                consigna
notebooks/           tp2_clasificacion.ipynb  ← se trabaja acá
src/tp2/
  config.py          rutas, semilla, k, columnas, métricas candidatas
  data.py            carga, split estratificado, submuestreo
  features.py        FeatureEngineer + ColumnTransformer (dentro del Pipeline → sin leakage)
  models.py          NB / SVM / KNN / RF + grillas de hiperparámetros
  evaluation.py      CV k-fold, curvas de validación, reporte de test (cacheado)
  plots.py           gráficos con valores anotados
reports/figures/     figuras exportadas para la presentación
```

## Decisiones de diseño (sólo conceptos vistos en clase)

- **Leakage**: split train/test antes de todo; escalado/encoding se ajustan dentro del `Pipeline`
  en cada fold. `duration` se excluye (sólo se conoce después de la llamada).
- **Preprocesamiento por modelo**: SVM/KNN → z-score + one-hot; RF → one-hot sin escalar;
  Naive Bayes → gaussiana para numéricas y frecuencias con Laplace para categóricas (clase 6).
- **Desbalance (~11 % yes)**: k-fold estratificado, sin accuracy; métricas ROC-AUC (elegir modelo)
  y F1 con el umbral elegido en train (clase 4).

## Rendimiento

- `n_jobs=-1` en la CV, curvas de validación, KNN y RF.
- Resultados de CV y curvas **cacheados en disco** (`.cache/`, `joblib.Memory`). La caché se
  invalida sola si cambia el código de `src/tp2` (salvo `plots.py`).
- SVM con kernel RBF escala ~O(n²): `cache_size=2000` y curva de validación sobre una submuestra
  estratificada (`data.stratified_subsample`).
