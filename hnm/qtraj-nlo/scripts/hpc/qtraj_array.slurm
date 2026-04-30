#!/bin/bash
# Array over NARRAY shards; each node runs P parallel qtraj workers.
#SBATCH -A m5098
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 06:00:00
#SBATCH -J qtraj_array
#SBATCH --array=0-$((${NARRAY:-64}-1))
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=64        # P: parallel workers/node (safe start; tune up to 128)
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH -o logs/%x.%A_%a.out
#SBATCH -e logs/%x.%A_%a.err

set -euo pipefail
umask 002

# ---- knobs ----
P="${SLURM_NTASKS_PER_NODE}"        # parallel workers per node
NQTRAJ="${NQTRAJ:-20}"              # quantum trajectories per temperature file
# ----------------

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
SCRATCH_DIR="${SCRATCH:?SCRATCH is not set}"
JOBID="${SLURM_ARRAY_JOB_ID}"
TASKID="${SLURM_ARRAY_TASK_ID}"
NARRAY="${NARRAY:?set via --export or env}"
TAR_PATH="${TAR_PATH:?set via --export or env}"

# Guard: with --hint=nomultithread, keep P <= 128 physical cores on Perlmutter CPU
if (( P > 128 )); then
  echo "ERROR: --ntasks-per-node=$P exceeds 128 physical cores with --hint=nomultithread."
  echo "Reduce P to <=128 or remove --hint=nomultithread."
  exit 1
fi

# Manifests dir from index step (most recent)
MANIDIR="$(ls -d "$SUBMIT_DIR"/manifests_* 2>/dev/null | sort | tail -n1)"
SHARD_LIST="$MANIDIR/shard_$(printf "%02d" "$TASKID").list"
[[ -f "$SHARD_LIST" ]] || { echo "Missing shard list: $SHARD_LIST"; exit 2; }

WORKDIR="$SCRATCH_DIR/qtraj_${JOBID}_${TASKID}"
OUTBASE="$SUBMIT_DIR/output_${JOBID}"
OUTDIR="$OUTBASE/task_${TASKID}"
mkdir -p "$WORKDIR" "$OUTDIR" "$SUBMIT_DIR/logs"

# Stripe only on SCRATCH
if [[ "$WORKDIR" == /pscratch/* || "$WORKDIR" == /global/cfs/* ]]; then
  command -v lfs >/dev/null 2>&1 && lfs setstripe -c 8 "$WORKDIR" || true
fi

# Stage small inputs + binary
module -q load fftw 2>/dev/null || module -q load fftw3 2>/dev/null || true
mkdir -p "$WORKDIR/input/runset" "$WORKDIR/out"
cp -v "$SUBMIT_DIR/qtraj" "$WORKDIR/"
cp -v "$SUBMIT_DIR/input/params.txt" "$WORKDIR/input/"
cp -v "$SUBMIT_DIR"/input/basisfunctions*.tsv "$WORKDIR/input/" 2>/dev/null || true
chmod +x "$WORKDIR/qtraj"

# Extract ONLY this shard’s members into $SCRATCH
echo "Extracting $(wc -l < "$SHARD_LIST") files for shard $TASKID"
tar -xzf "$TAR_PATH" --warning=no-timestamp -C "$WORKDIR/input/runset" --files-from "$SHARD_LIST"

# Build workload list (relative paths)
mapfile -t FILES < <(cd "$WORKDIR/input/runset" && find . -type f -print | sed 's|^\./||' | sort)
echo "Shard $TASKID has ${#FILES[@]} files"

# Keep single-threaded math libs (avoid hidden oversubscription)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# Worker: two initL passes per file; bind to physical cores; prefer local NUMA memory
run_one() {
  local rel="$1" abs="$WORKDIR/input/runset/$rel"
  srun --exclusive -N1 -n1 -c1 --cpu-bind=cores --mem-bind=local \
    "$WORKDIR/qtraj" -initL 0 -nTrajectories "$NQTRAJ" -temperatureEvolution 2 -temperatureFile "$abs" >/dev/null 2>&1
  srun --exclusive -N1 -n1 -c1 --cpu-bind=cores --mem-bind=local \
    "$WORKDIR/qtraj" -initL 1 -nTrajectories "$NQTRAJ" -temperatureEvolution 2 -temperatureFile "$abs" >/dev/null 2>&1
}

# Launch up to P in parallel
pids=(); i=0
for f in "${FILES[@]}"; do
  run_one "$f" &
  pids+=($!)
  (( ++i % P == 0 )) && { wait "${pids[@]}"; pids=(); }
done
((${#pids[@]})) && wait "${pids[@]}"

# Aggregate this shard’s outputs -> one gz (robust to large file counts)
if find "$WORKDIR" -type f -name 'ratios.tsv' | read -r _; then
  find "$WORKDIR" -type f -name 'ratios.tsv' -print0 \
    | xargs -0 cat | gzip > "$WORKDIR/datafile_task${TASKID}.gz"
  rsync -av "$WORKDIR/datafile_task${TASKID}.gz" "$OUTDIR/"
else
  echo "WARNING: No ratios.tsv found for shard $TASKID"
fi

# (Optional) keep raw outputs (can be large)
rsync -av --ignore-errors "$WORKDIR"/output*/ "$OUTDIR/outputs/" 2>/dev/null || true

echo "Shard $TASKID done. Output in: $OUTDIR"

