# Resumen baseline predictores — evidencia para decisión RF + GB

**Referencia congelada:** `SNAPSHOT_20260511_MN5.json` (jobs SLURM **40400325** Fase A, **40400326** Fase B).  
**Regenerar desde CSV del cluster:** `python scripts/export_baseline_summary.py` (véase `README.md`).

## Decisión apoyada por datos

Centrar el **refinado (tuning moderado)** en **Random Forest** y **HistGradient Boosting** porque en las tres tareas son **1.º y 2.º** en la métrica principal de test, con **brecha clara** respecto a lineales, MLP y (en Fase B) árbol aislado.

## Fase A — regresión coste (orden por test MAE ↓)

| Rank | Familia | test MAE | test RMSE | test R² | fit (s) |
|------|---------|----------|------------|----------|---------|
| 1 | random_forest | 67.37 | 189.51 | 0.9966 | 1.44 |
| 2 | decision_tree | 81.35 | 236.26 | 0.9947 | 0.06 |
| 3 | gradient_boosting | 111.86 | 229.97 | 0.9950 | 1.05 |
| 4 | mlp | 278.19 | 596.78 | 0.9664 | 17.37 |
| 5 | linear | 822.87 | 1177.01 | 0.8692 | 0.01 |
| 6 | lasso | 822.88 | 1177.01 | 0.8692 | 1.08 |
| 7 | ridge | 856.93 | 1195.54 | 0.8651 | 0.005 |

*Nota:* en inferencia sobre test, **RF** tuvo `predict_time_test_s` ~**0.091 s** frente a **GB** ~**0.011 s** (2656 filas) — trade-off error vs latencia de predicción a mencionar en memoria.

## Fase B — ex ante (orden por test ROC-AUC ↓)

| Rank | Familia | ROC-AUC | F1 | Accuracy | fit (s) |
|------|---------|---------|-----|----------|---------|
| 1 | random_forest | 0.8916 | 0.8087 | 0.8007 | 1.26 |
| 2 | gradient_boosting | 0.8859 | 0.8039 | 0.7927 | 1.05 |
| 3 | mlp | 0.8388 | 0.7737 | 0.7593 | 6.87 |
| 4 | decision_tree | 0.8198 | 0.7871 | 0.7709 | 0.05 |
| 5 | logistic_l1 | 0.7878 | 0.7282 | 0.7042 | 1.30 |
| 6 | logistic | 0.7854 | 0.7266 | 0.7024 | 0.94 |
| 7 | logistic_l2 | 0.7854 | 0.7266 | 0.7024 | 0.93 |

## Fase B — post hoc (orden por test ROC-AUC ↓)

| Rank | Familia | ROC-AUC | F1 | Accuracy | fit (s) |
|------|---------|---------|-----|----------|---------|
| 1 | random_forest | 0.9358 | 0.8695 | 0.8617 | 1.37 |
| 2 | gradient_boosting | 0.9336 | 0.8645 | 0.8572 | 1.13 |
| 3 | mlp | 0.8954 | 0.8226 | 0.8158 | 6.47 |
| 4 | logistic_l1 | 0.8934 | 0.8313 | 0.8216 | 4.38 |
| 5 | logistic | 0.8681 | 0.7993 | 0.7905 | 0.97 |
| 6 | logistic_l2 | 0.8681 | 0.7993 | 0.7905 | 0.94 |
| 7 | decision_tree | 0.8404 | 0.8159 | 0.8105 | 0.11 |

## Caveats metodológicos (una frase en memoria)

- `logistic` y `logistic_l2` son **redundantes** (misma configuración efectiva).
- Fase B en esta corrida: **~2248** ejemplos de test → coherente con **~15 k filas** (config por defecto); para tabla **solo trazas buenas (7500)** hay que repetir train con `configs/phase_b_retrain_good_traces_only.yaml` y actualizar snapshot.
- Avisos sklearn en logs (Lasso / convergencia logística) no invalidan el **ranking** baseline; sí limitan interpretación fina de coeficientes hasta escalar o subir `max_iter`.
