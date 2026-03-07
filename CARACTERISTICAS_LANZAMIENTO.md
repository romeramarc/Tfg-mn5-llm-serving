# CARACTERÍSTICAS DEL LANZAMIENTO - FASE 1

## QUÉ ES

Un **script SBATCH** (Slurm Batch) que define:
- **Partición:** acc (nodos H100)
- **Cuenta:** bsc98
- **QoS:** acc_bsccs
- **Recursos:** 1 GPU H100, 20 CPUs, 18 horas
- **Acción:** Ejecuta benchmarking de 3 modelos Qwen secuencialmente

---

## LOS 3 JOBS (en SECUENCIA dentro del mismo sbatch)

### Job 1: TEACHER (14B)
```
Modelo:     Qwen/Qwen2.5-14B-Instruct
GPU:        1× H100
CPU:        20 cores
Duración:   ~20-30 min (estimado)

Tareas:
  1. Quality evaluation (GSM8K + MATH-500)
  2. Routing evaluation
  3. Throughput benchmark (×3 repeticiones)
  4. Online load benchmark (×3 repeticiones)
```

### Job 2: STUDENT MID (7B)
```
Modelo:     Qwen/Qwen2.5-7B-Instruct
GPU:        1× H100 (mismo)
CPU:        20 cores (mismo)
Duración:   ~15-20 min (estimado)

Tareas:
  1. Quality evaluation (GSM8K + MATH-500)
  2. Throughput benchmark (×3 repeticiones)
  3. Online load benchmark (×3 repeticiones)
```

### Job 3: STUDENT SMALL (1.5B)
```
Modelo:     Qwen/Qwen2.5-1.5B-Instruct
GPU:        1× H100 (mismo)
CPU:        20 cores (mismo)
Duración:   ~10-15 min (estimado)

Tareas:
  1. Quality evaluation (GSM8K + MATH-500)
  2. Throughput benchmark (×3 repeticiones)
  3. Online load benchmark (×3 repeticiones)
```

---

## TIEMPO TOTAL

```
Teacher:      ~25-30 min
Student 7B:   ~15-20 min
Student 1.5B: ~10-15 min
─────────────────────────
TOTAL:        ~50-65 min

Histórico confirma: Job anterior (36980381) → 76 min
Límite solicitado: 18 horas
→ ✅ Suficiente margen
```

---

## RECURSOS SOLICITADOS (SBATCH directives)

```bash
#SBATCH --job-name=fase1-baselines
#SBATCH --partition=acc              ← H100 GPU nodes
#SBATCH --account=bsc98              ← Tu cuenta
#SBATCH --qos=acc_bsccs              ← Queue type (SATURADA)
#SBATCH --gres=gpu:1                 ← 1 GPU
#SBATCH --ntasks=1                   ← 1 task
#SBATCH --cpus-per-task=20           ← 20 CPU cores
#SBATCH --time=18:00:00              ← 18 hours max
#SBATCH --output=logs/fase1-baselines-%j.out
#SBATCH --error=logs/fase1-baselines-%j.err
```

---

## OUTPUTS GENERADOS

Por cada modelo:
```
results/quality/quality-{MODELO}-{TIMESTAMP}/
results/throughput/throughput-{MODELO}-{TIMESTAMP}/ (×3)
results/online/online-{MODELO}-{TIMESTAMP}/ (×3)
results/routing/routing-teacher-{TIMESTAMP}/
```

---

## ESTADO ACTUAL

```
Job ID:      37344851 (fase1)
Job ID:      37344852 (fase2 - pending on 37344851)
Job ID:      37344853 (fase3 - pending on 37344852)

Duración PENDING: 6+ horas
Reason: Priority
Causa: QoS acc_bsccs saturada (GrpJobs=N(0))
```

---

## EL PROBLEMA RESUMIDO

```
┌─────────────────────────────────────┐
│ Tu sbatch pide:                     │
│ - 1 GPU H100 ✅ disponibles        │
│ - 20 CPU cores ✅ disponibles      │
│ - partition acc ✅ tienes permiso   │
│ - QoS acc_bsccs ❌ LLENA (0 slots)  │
└─────────────────────────────────────┘

Resultado: Job entra en la cola pero NO SE EJECUTA
porque no hay espacio en ese QoS específico,
aunque sí lo hay en partition acc en general.
```

---

## SOLUCIÓN PROPUESTA

**Cambiar de:**
```bash
--partition=acc --qos=acc_bsccs --time=18:00:00
```

**A:**
```bash
--partition=gpp --time=08:00:00
```

**Razón:** gpp tiene miles de nodos libres, job empezará en minutos en lugar de estar pending indefinidamente.

---

**RESUMEN PARA TU PROFE:**

"Lanzo un sbatch **fase1_baselines.sbatch** que secuencialmente benchmarkea 3 modelos Qwen (14B teacher, 7B y 1.5B students) en 1 H100 GPU. Pide 18 horas en partition acc/qos acc_bsccs. El job anterior igual ejecutó exitosamente en 76 minutos. Ahora está PENDING 6+ horas porque ese QoS está saturado (GrpJobs=0), aunque partition acc tiene recursos libres. Recomiendo cambiar a partition gpp donde hay espacio y bajar tiempo a 8h (más que suficiente para 76 min de ejecución real)."
