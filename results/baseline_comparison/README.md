# Evidencia baseline — predictores (Fase A + Fase B)

Aquí vive el **resumen versionado** que justifica en la memoria la decisión de **refinar solo Random Forest y Gradient Boosting** frente al resto de familias.

## Fuente de verdad en el cluster

Tras cada `sbatch slurm/phase_a_train.sbatch` / `phase_b_train.sbatch`, los CSV comparativos se generan en:

| Experimento | Ruta (relativa al repo) |
|-------------|-------------------------|
| Coste servicio (regresión) | `results/phase_a/predictors/service_cost_baseline_comparison.csv` |
| Calidad ex ante (clasificación) | `results/phase_b/predictors/quality_ex_ante_baseline_comparison.csv` |
| Calidad post hoc (clasificación) | `results/phase_b/predictors/quality_post_hoc_baseline_comparison.csv` |

Cada fila = una familia; columnas = métricas test/val, tiempos de fit/predict y rutas a `metrics.json` del modelo.

## Congelar evidencia en git (PC de trabajo)

1. Copia los tres CSV desde BSC a las rutas anteriores (mismo layout que en el cluster), **o** pasa rutas explícitas al script.
2. En la raíz del repo:

```bash
python scripts/export_baseline_summary.py
```

Genera:

- `summary_<UTC>.json` — payload completo (rankings, RF vs GB, narrativa).
- `LATEST.json` — copia del último run (referencia rápida).
- `SUMMARY_FOR_MEMORIA.md` — tablas listas para adaptar a LaTeX.

3. `git add results/baseline_comparison/*.json results/baseline_comparison/*.md` y commit con mensaje descriptivo.

## Snapshot de referencia (11-may-2026, MN5)

El fichero `SNAPSHOT_20260511_MN5.json` congela el resultado del par de jobs **40400325** (Fase A) y **40400326** (Fase B) tal como constaba en los CSV del cluster. Si vuelves a exportar desde CSV actualizados, el script puede divergir; el snapshot queda como **ancla documental** de esa corrida.
