#!/bin/bash
# =============================================================================
#  ÚNICO punto de entrada recomendado — pipeline completo KD + eval
# =============================================================================
#  Objetivo (feedback profe): priorizar mejora de calidad en GSM8K frente al
#  teacher de referencia (92,41 %), sin olvidar MATH. Los hiperparámetros y el
#  oversampling del TRAIN están en configs/distill.yaml (Exp. 4).
#
#  Un solo comando desde la raíz del repo (BSC login):
#      bash slurm/launch.sh
#
#  Implementación: delega en launch_distill_v4.sh (5 jobs SLURM encadenados).
# =============================================================================

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "${ROOT}/slurm/launch_distill_v4.sh"
