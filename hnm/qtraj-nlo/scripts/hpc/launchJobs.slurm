#!/bin/bash -l
# Perlmutter CPU — qtraj job array
# 1 array element = 1 full CPU node (128 cores, no SMT)
# Pre-extract slice once -> then 128 single-thread workers fan out with srun.
# Outputs isolated per element; final datafile.gz is lock-appended (no mixing).

#SBATCH -A m5098
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J qtraj
#SBATCH -t 02:00:00                 # target 60–120 min once you confirm pacing
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --ntasks-per-socket=64      # Perlmutter CPU: 2 sockets × 64 cores
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH -o logs/%x.%A_%a.out
#SBATCH -e logs/%x.%A_%a.err

set -euo pipefail
trap 'echo "[trap] status=$? on $(date)"; exit $?' EXIT

# ---------------- user knobs (override via sbatch --export or env) ------------
: "${NPTRAJ_TOTAL:=250000}"        # total files across the WHOLE array job
: "${CHUNK_NP:=3072}"              # files handled by THIS array element (~60–90 min)
: "${NQTRAJ:=20}"                  # quantum trajectories per NPTRAJ
: "${SEED:=42}"                    # deterministic shuffle for list creation
: "${DO_AGGREGATE:=1}"             # write ratios_<AID>.tsv.gz and combined logs
: "${STRIPE_OUT:=0}"               # 1 => set wide striping on OUT_TOP (optional)

# Inputs (default: from submit dir). Best practice: keep tarball on $PSCRATCH.
: "${TRAJ_TGZ:=${SLURM_SUBMIT_DIR}/input/runset/trajectories.tgz}"
: "${PARAMS_TXT_SRC:=${SLURM_SUBMIT_DIR}/input/params.txt}"
: "${BASIS_GLOB:=${SLURM_SUBMIT_DIR}/input/basisfunction*.tsv}"
: "${QTRAJ_BIN_SRC:=${SLURM_SUBMIT_DIR}/qtraj}"

[[ -r "$TRAJ_TGZ" ]]       || { echo "ERROR: missing $TRAJ_TGZ"; exit 2; }
[[ -x "$QTRAJ_BIN_SRC" ]]  || { echo "ERROR: missing/exec $QTRAJ_BIN_SRC"; exit 2; }
[[ -r "$PARAMS_TXT_SRC" ]] || { echo "ERROR: missing $PARAMS_TXT_SRC"; exit 2; }

# ---------------- env & binding ----------------------------------------------
module load PrgEnv-gnu gcc-native/12.3 2>/dev/null || true
module unload gpu cudatoolkit 2>/dev/null || true
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SLURM_CPU_BIND=cores

# ---------------- array-scoped paths -----------------------------------------
ARR_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
AID="${SLURM_ARRAY_TASK_ID:-0}"
RUNROOT="${PSCRATCH:-/pscratch/sd/${USER:0:1}/$USER}/qtraj_${ARR_ID}"
STAGE="${RUNROOT}/stage"
LISTS="${RUNROOT}/lists"
WORKDIR="${RUNROOT}/work/elem_${AID}"
OUT_TOP="${SLURM_SUBMIT_DIR}/output-${ARR_ID}"

umask 027
mkdir -p "${STAGE}" "${LISTS}" "${WORKDIR}/input/runset" "${WORKDIR}/output" "${RUNROOT}/out" "${RUNROOT}/logs" "${OUT_TOP}"

# Optionally stripe the output directory for faster single-file append (Lustre)
if (( STRIPE_OUT )); then
  command -v lfs >/dev/null 2>&1 && lfs setstripe -c -1 "${OUT_TOP}" || true
fi
# Also strongly recommended: keep TRAJ_TGZ on $PSCRATCH for read bandwidth.  # :contentReference[oaicite:4]{index=4}

# ---------------- master list build (once per array job) ----------------------
exec 9> "${LISTS}/.lock"
flock -x 9
if [[ ! -s "${LISTS}/all.txt" ]]; then
  echo "[prep] indexing $TRAJ_TGZ ..."
  tar -tzf "$TRAJ_TGZ" | grep -E '\.txt$' | shuf --random-source=<(yes "$SEED") > "${LISTS}/all.txt"
fi
if [[ ! -s "${LISTS}/target_${NPTRAJ_TOTAL}.txt" ]]; then
  head -n "$NPTRAJ_TOTAL" "${LISTS}/all.txt" > "${LISTS}/target_${NPTRAJ_TOTAL}.txt"
fi
SLICES=$(( (NPTRAJ_TOTAL + CHUNK_NP - 1) / CHUNK_NP ))
if (( AID >= SLICES )); then
  echo "[info] AID ${AID} >= SLICES ${SLICES}; nothing to do."
  flock -u 9
  exit 0
fi
flock -u 9

START=$(( AID * CHUNK_NP + 1 ))
END=$(( START + CHUNK_NP - 1 ))
(( END > NPTRAJ_TOTAL )) && END="$NPTRAJ_TOTAL"
SLICE="${LISTS}/slice_${AID}.txt"
sed -n "${START},${END}p" "${LISTS}/target_${NPTRAJ_TOTAL}.txt" > "$SLICE"
NUM_IN_SLICE=$(wc -l < "$SLICE" || echo 0)
(( NUM_IN_SLICE>0 )) || { echo "[info] empty slice ${AID}; exit."; exit 0; }
echo "[info] slice ${AID}: ${NUM_IN_SLICE} files (${START}..${END})"

# ---------------- stage small inputs & sbcast tiny binary ---------------------
cp -f "$PARAMS_TXT_SRC" "${STAGE}/params.txt"
cp -f $BASIS_GLOB "${STAGE}/" 2>/dev/null || true
LOCAL_EXE="/tmp/${USER}_qtraj_${ARR_ID}"
sbcast "$QTRAJ_BIN_SRC" "$LOCAL_EXE"   # localize executable to node /tmp (faster)  # :contentReference[oaicite:5]{index=5}
chmod +x "$LOCAL_EXE"

# ---------------- one-time pre-extract THIS slice -----------------------------
echo "[$(date)] extracting ${NUM_IN_SLICE} files ..."
tar -xzf "$TRAJ_TGZ" --warning=no-timestamp --files-from "$SLICE" -C "${WORKDIR}/input/runset"
echo "[$(date)] extraction done."

# ---------------- fan-out compute (128 ranks, 1 per core) ---------------------
export PARAMS_TXT="${STAGE}/params.txt"
export NQTRAJ
T0=$(date +%s)

srun --distribution=block:block --cpu-bind=cores --kill-on-bad-exit=1 -n "$SLURM_NTASKS" bash -lc '
  set -euo pipefail
  R=$SLURM_NTASKS
  lines=$(wc -l < "'"$SLICE"'")
  i=$(( SLURM_PROCID + 1 ))
  while (( i<=lines )); do
    traj_rel=$(sed -n "${i}p" "'"$SLICE"'")
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
printf "[timing] slice %d: %d files in %ds; ~%.2f files/min\n" "$AID" "$NUM_IN_SLICE" "$DT" "$TPM"

# ---------------- per-element aggregation (no mixing) -------------------------
if (( DO_AGGREGATE )); then
  find "${WORKDIR}/output" -type f -name "ratios.tsv" -print0 \
    | xargs -0 cat | gzip > "${RUNROOT}/out/ratios_${AID}.tsv.gz" || true

  find "${WORKDIR}/output" -type f -name "*.log" -print0 \
    | xargs -0 cat > "${RUNROOT}/logs/qtraj_${AID}.log" || true

  # Lock-append into a single concatenated gzip (zcat supports concatenation)
  exec 8>> "${OUT_TOP}/.append.lock"
  flock -x 8
  cat "${RUNROOT}/out/ratios_${AID}.tsv.gz" >> "${OUT_TOP}/datafile.gz"
  flock -u 8

  echo "[info] appended ratios_${AID}.tsv.gz -> ${OUT_TOP}/datafile.gz"
fi

echo "[done] elem=${AID} outputs: ${WORKDIR}/output ; ${RUNROOT}/out/ratios_${AID}.tsv.gz"

