#!/usr/bin/env bash
# scripts/git_add_official_results_mar9.sh
# ─────────────────────────────────────────
# Añade al índice de git todos los artefactos "oficiales" del 9 mar 2026
# (calidad, throughput, online, agregado, generación teacher de esa oleada).
#
# NO incluye los directorios merged-qwen2.5-* (pesos); están en .gitignore.
# Ejecutar desde la RAÍZ del repo:
#   bash scripts/git_add_official_results_mar9.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== Repo: $ROOT"
echo "== Añadiendo resultados oficiales 2026-03-09 (métricas y configs)"
echo ""

add_if_exists() {
  local p="$1"
  if [[ -e "$p" ]]; then
    git add -- "$p"
    echo "  + $p"
  else
    echo "  (omitido, no existe: $p)"
  fi
}

# Resumen agregado
add_if_exists "results/summary_all_models.csv"

# Calidad: todas las runs con timestamp 20260309
shopt -s nullglob
for d in results/quality/quality-*-20260309*; do
  add_if_exists "$d"
done

# Throughput
for d in results/throughput/throughput-*-20260309*; do
  add_if_exists "$d"
done

# Online
for d in results/online/online-*-20260309*; do
  add_if_exists "$d"
done

# Generación teacher de la oleada del 9 (configs / metadatos; sin pesos enormes si solo hay yaml)
add_if_exists "results/distill/teacher-gen-20260309T081808Z"

# Logs explícitos de post-eval del 9 mar (jobs 37554282 / 37554283)
for f in logs/posteval-7b-37554282.out logs/posteval-7b-37554282.err \
         logs/posteval-1.5b-37554283.out logs/posteval-1.5b-37554283.err; do
  add_if_exists "$f"
done

shopt -u nullglob

# Papelera opcional: teacher_outputs.jsonl (puede ser ~12MB; descomenta si quieres versionarlo)
# git add -- results/distill/teacher_outputs.jsonl

echo ""
echo "== Estado (staged):"
git status --short

echo ""
echo "Siguiente paso (revisa el listado antes de commit):"
echo "  git commit -m \"Add official Mar 9 2026 experiment results (quality, throughput, online, summary)\""
echo "  git push origin main"
