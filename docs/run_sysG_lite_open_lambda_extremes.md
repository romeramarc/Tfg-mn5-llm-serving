# RouteCascade lite — extremos de λ en el router (`sysG_lite_open_l0`, `sysG_lite_open_l5e4`)

Dos runs con la **misma calibración** que `sysG_lite_open` (umbrales post-hoc por rung, `max_attempts=2`, teacher en candidatos). Solo cambia **`cost_weight_lambda`** del router ex-ante:

| ID | λ | Router maximiza | Punto de referencia |
|----|---|-----------------|---------------------|
| **sysG_lite_open_l0** | `0` | Calidad (`U = Q̂`) | Análogo a `sysE_l0` pero con cascada lite calibrada |
| **sysG_lite_open_l5e4** | `5×10⁻⁴` | Servicio (casi todo a 1.5B) | Análogo a `sysE_l5e4` con misma cascada |
| sysG_lite_open (ya ejecutado) | `5×10⁻⁵` | Trade-off intermedio | — |

La **cascada post-hoc** (umbrales 0.50 / 0.30 / 0.25 / 1.0, parseable 0.45) es idéntica en los tres.

## Proceso en BSC (MN5)

```bash
cd /gpfs/scratch/bsc98/tbsc381408/Tfg-mn5-llm-serving   # PROJECT_DIR
git pull
export EVAL_CONFIG=configs/routing_eval_holdout_v2_routing_real.yaml
export PROMPT_POOL=results/routing_eval_holdout/prompt_pool.jsonl

# 1) Servidores vLLM (si no están ya levantados)
bash slurm/launch_eval_holdout.sh servers
# Esperar results/routing/endpoints/*.url

# 2) Preflight + dos evals (NO usar `clients` sin filtrar: lanzaría todo el YAML)
python scripts/preflight_eval_holdout.py --config "${EVAL_CONFIG}"

for sid in sysG_lite_open_l0 sysG_lite_open_l5e4; do
  sbatch --job-name="eval-${sid}" --time=14:00:00 \
    --export=ALL,SYSTEM_ID="${sid}",PROJECT_DIR="${PWD}",\
EVAL_CONFIG="${EVAL_CONFIG}",PROMPT_POOL="${PROMPT_POOL}",ENDPOINT_WAIT_S=900 \
    slurm/eval_holdout.sbatch
done
```

Resultados bajo `results/routing_eval_holdout_v2_routing_real/<system_id>-routing_plus_cascade-<timestamp>/`.

## Qué comparar después

- **l0 vs lite_open (5e-5):** ¿cuánto gana calidad la cascada lite si el router ya apunta a rungs grandes?
- **l5e4 vs lite_open:** mismo coste de router “barato”, distinto post-hoc (vs `sysE_l5e4` sin calibración por rung).
