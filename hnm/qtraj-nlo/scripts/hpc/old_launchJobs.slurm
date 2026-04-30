#!/bin/bash
#===============================================================================
# Perlmutter CPU — Efficient qtraj via Slurm ARRAY (1 array task = 1 full node)
#   • Inside each array task: 128 single-threaded workers (1/core, no SMT)
#   • Executable staged to /tmp with sbcast; hot I/O in $PSCRATCH
#   • Each array task processes CHUNK_NP trajectories; prints timing/throughput
#===============================================================================

#SBATCH -A m5098
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J qtraj
#SBATCH -t 48:00:00                # set conservatively; tighten after a timing run
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128      # 128 physical cores / Perlmutter CPU node
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread       # bind to physical cores (no SMT)
#SBATCH -o logs/%x.%A_%a.out
#SBATCH -e logs/%x.%A_%a.err
# Submit with: sbatch -N 1 --array=0-<S-1>%<CONC> qtraj_array.slurm

set -euo pipefail

# ----------------------- knobs (override via env) -----------------------------
: "${NPTRAJ_TOTAL:=250000}"        # total physical trajectories overall
: "${CHUNK_NP:=4096}"              # trajectories handled by *this* array task
: "${NQTRAJ:=20}"                  # quantum trajectories per physical traj
: "${SEED:=42}"                    # deterministic shuffle seed
: "${DO_AGGREGATE:=1}"             # 1=write per-slice ratios_${AID}.tsv.gz

# inputs (defaults assume submit from repo root)
: "${TRAJ_TGZ:=$SLURM_SUBMIT_DIR/input/runset/trajectories.tgz}"
: "${PARAMS_TXT_SRC:=$SLURM_SUBMIT_DIR/input/params.txt}"
: "${BASIS_GLOB:=$SLURM_SUBMIT_DIR/input/basisfunction*.tsv}"
: "${QTRAJ_BIN_SRC:=$SLURM_SUBMIT_DIR/qtraj}"

[[ -r "$TRAJ_TGZ" ]]       || { echo "ERROR: missing $TRAJ_TGZ"; exit 2; }
[[ -r "$QTRAJ_BIN_SRC" ]]  || { echo "ERROR: missing $QTRAJ_BIN_SRC"; exit 2; }
[[ -r "$PARAMS_TXT_SRC" ]] || { echo "ERROR: missing $PARAMS_TXT_SRC"; exit 2; }

# ----------------------- work in PSCRATCH (NERSC best practice) ---------------
RUNROOT="${PSCRATCH:-/pscratch/sd/${USER:0:1}/$USER}/qtraj_${SLURM_JOB_ID}"
mkdir -p "$RUNROOT"/{lists,stage,work,logs,out}
echo "RUNROOT=$RUNROOT"

# ----------------------- index once, deterministic shuffle --------------------
LIST_ALL="$RUNROOT/lists/all.txt"
if [[ ! -s "$LIST_ALL" ]]; then
  echo "[prep] indexing .txt members in $TRAJ_TGZ ..."
  tar -tzf "$TRAJ_TGZ" | grep -E '\.txt$' | shuf --random-source=<(yes "$SEED") > "$LIST_ALL"
fi
LIST_TARGET="$RUNROOT/lists/target_${NPTRAJ_TOTAL}.txt"
[[ -s "$LIST_TARGET" ]] || head -n "$NPTRAJ_TOTAL" "$LIST_ALL" > "$LIST_TARGET"

# ----------------------- compute this array slice -----------------------------
AID=${SLURM_ARRAY_TASK_ID:-0}
START=$(( AID * CHUNK_NP + 1 ))
END=$(( START + CHUNK_NP - 1 ))
SLICE="$RUNROOT/lists/slice_${AID}.txt"
sed -n "${START},${END}p" "$LIST_TARGET" > "$SLICE"
NUM_IN_SLICE=$(wc -l < "$SLICE" || echo 0)
(( NUM_IN_SLICE>0 )) || { echo "[info] empty slice; exiting."; exit 0; }
echo "[info] slice $AID: $NUM_IN_SLICE trajectories ($START..$END)"

# ----------------------- stage exec to /tmp (sbcast) --------------------------
LOCAL_EXE="/tmp/${USER}_qtraj_${SLURM_JOB_ID}"
sbcast "$QTRAJ_BIN_SRC" "$LOCAL_EXE"
chmod +x "$LOCAL_EXE"
cp -f "$PARAMS_TXT_SRC" "$RUNROOT/stage/params.txt"
cp -f $BASIS_GLOB "$RUNROOT/stage/" 2>/dev/null || true

WORKDIR="$RUNROOT/work/slice_${AID}"
mkdir -p "$WORKDIR/input/runset" "$WORKDIR/output"

# ----------------------- binding & env ----------------------------------------
export SLURM_CPU_BIND=cores        # 1 task → 1 physical core
export OMP_NUM_THREADS=1
export NQTRAJ
export PARAMS_TXT="$RUNROOT/stage/params.txt"

# ----------------------- timed fan-out (128 workers) --------------------------
T0=$(date +%s)
srun -n "$SLURM_NTASKS" bash -lc '
  set -euo pipefail
  R=$SLURM_NTASKS
  lines=$(wc -l < "'"$SLICE"'")
  i=$(( SLURM_PROCID + 1 ))
  while (( i<=lines )); do
    traj_rel=$(sed -n "${i}p" "'"$SLICE"'")
    mkdir -p "'"$WORKDIR"'/input/runset/$(dirname "$traj_rel")"
    tar --warning=no-timestamp -xzf "'"$TRAJ_TGZ"'" -C "'"$WORKDIR"'/input/runset" "$traj_rel"
    tfile="'"$WORKDIR"'/input/runset/$traj_rel"

    outd="'"$WORKDIR"'/output/$(basename "$tfile" .txt)"
    mkdir -p "$outd"

    "'"$LOCAL_EXE"'" -initL 0 -nTrajectories "'"$NQTRAJ"'" -temperatureEvolution 2 \
      -temperatureFile "$tfile" -params "'"$PARAMS_TXT"'" > "$outd/l0.log" 2>&1
    "'"$LOCAL_EXE"'" -initL 1 -nTrajectories "'"$NQTRAJ"'" -temperatureEvolution 2 \
      -temperatureFile "$tfile" -params "'"$PARAMS_TXT"'" > "$outd/l1.log" 2>&1

    i=$(( i + R ))
  done
'
T1=$(date +%s); DT=$((T1-T0)); ((DT==0)) && DT=1
TPM=$(awk -v n="$NUM_IN_SLICE" -v s="$DT" 'BEGIN{print (n/(s/60.0))}')
printf "[timing] slice %d: %d trajectories in %ds; ~%.2f traj/min\n" \
  "$AID" "$NUM_IN_SLICE" "$DT" "$TPM"

# ----------------------- per-slice aggregation (optional) ---------------------
if (( DO_AGGREGATE )); then
  find "$WORKDIR/output" -type f -name "ratios.tsv" -print0 \
    | xargs -0 cat | gzip > "$RUNROOT/out/ratios_${AID}.tsv.gz" || true
  find "$WORKDIR/output" -type f -name "*.log" -print0 \
    | xargs -0 cat > "$RUNROOT/logs/qtraj_${AID}.log" || true
fi

echo "[done] outputs at: $WORKDIR/output"

