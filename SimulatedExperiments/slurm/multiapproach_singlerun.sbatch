#!/bin/bash
#SBATCH --job-name=geoseries_full
#SBATCH --partition=normal
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64       # 2 x 64 AMD CPUs with 64 cores each per node. GP usually runs faster on one socket so trying 64 here.
#SBATCH --mem=256G               # 256GB is for hte whole job on that node. if ntasks=1 then this is effectively per task.
#SBATCH --hint=nomultithread            # prefer 1 hardware thread per core - better for heavy floating point workloads
#SBATCH --exclusive                 # avoid noisy neighbors (optional but recommended)
#SBATCH --time=12:00:00
#SBATCH --array=0                             # --array=0-9 for 10 independent runs. Add %num (e.g., %5) to limit concurrent tasks if desired
#SBATCH --output=logs/%x_%A_%a.out          # %x is job name. %A jobid (array's master job), %a array index - %j is jobid when not using an array
#SBATCH --error=logs/%x_%A_%a.err           # %x is job name. %A jobid (array's master job), %a array index - %j is jobid when not using an array

set -euo pipefail
set -x

# Paths
PROJ="/nsls2/users/acorrao/GitHub/pub-Corrao_2025_09"
SCRDIR="$PROJ/SimulatedExperiments/slurm"
PYFILE="$SCRDIR/full_campaign_options.py"
DATASRC="$PROJ/Data/simulatedwafer_datasets"   # shared FS source

mkdir -p "$SCRDIR/logs"
cd "$SCRDIR"

# Ensure pixi is available on compute nodes
export PATH="$HOME/.pixi/bin:$PATH"
which pixi || { echo "ERROR: pixi not found on PATH"; exit 127; }
pixi --version

# ------------------------------
# Stage inputs to node-local disk
# ------------------------------
# Prefer Slurm-provided scratch; otherwise use /tmp/$SLURM_JOB_ID
NODE_SCRATCH="${SLURM_TMPDIR:-/tmp/$SLURM_JOB_ID}"
NODE_SIMDATA="$NODE_SCRATCH/simdata"
mkdir -p "$NODE_SIMDATA"

echo "Staging inputs from $DATASRC -> $NODE_SIMDATA"
# Copy only once per job; --ignore-existing avoids re-copying and reduces metadata ops
rsync -a --ignore-existing --info=stats1 "$DATASRC/" "$NODE_SIMDATA/"

# Point Python to fast local copy
export SIM_DATA_DIR="$NODE_SIMDATA"

# Also make temp dirs local so libs that spill tmp files are fast
export TMPDIR="$NODE_SCRATCH"
export MPLBACKEND=Agg # for matplotlib with no display (headless plotting)

# Thread caps (also enforced inside Python)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK}
export BLIS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Optional: clean up the staged inputs when the job exits
# trap 'rm -rf "$NODE_SIMDATA"' EXIT

# ------------------------------
# Run one replicate per array task
# ------------------------------
srun --cpu-bind=cores --mem-bind=local pixi run --manifest-path "$PROJ/pixi.toml" \
  python "$PYFILE" \
  --dataset complex \
  --total-meas 805 \
  --saved-data-pkl full \
  --approach geoseries \
  --campaign gs_full_0