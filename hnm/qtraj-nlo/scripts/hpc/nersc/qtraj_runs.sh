#!/bin/bash -l
# qtraj_runs.slurm — one bundle per CPU node, 128 ranks, node-local I/O
#
# Smoke (1 bundle on debug):
#   MANIFEST="$SCRATCH/qtraj_bundle_lists/manifest.txt"
#   sbatch -C cpu -q debug -t 00:30:00 \
#          --array=0-0 \
#          --export=ALL,MANIFEST="$MANIFEST",NQTRAJ=20,LOCALIZE_TAR=1,TASK_TIMEOUT=1200s,RETRIES=2 \
#          scripts/qtraj_runs.slurm
#
# Small array (first 8 bundles, 4 at a time):
#   NB=8
#   sbatch -C cpu -q regular -t 01:00:00 \
#          --array=0-$((NB-1))%4 \
#          --export=ALL,MANIFEST="$MANIFEST",NQTRAJ=20,LOCALIZE_TAR=1,TASK_TIMEOUT=1200s,RETRIES=2 \
#          scripts/qtraj_runs.slurm
#
# Production (all bundles; throttle with %NN per availability/policy):
#   NBUNDLES=$(wc -l < "$MANIFEST")
#   sbatch -C cpu -q regular -t 01:00:00 \
#          --array=0-$((NBUNDLES-1))%32 \
#          --export=ALL,MANIFEST="$MANIFEST",NQTRAJ=20,LOCALIZE_TAR=1,TASK_TIMEOUT=1200s,RETRIES=2 \
#          scripts/qtraj_runs.slurm

#SBATCH -A m5098
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -J qtraj
#SBATCH -N 1 --exclusive
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH -o logs/%x.%A_%a.out
#SBATCH -e logs/%x.%A_%a.err

set -euo pipefail
umask 002

# ---------------- User/config ----------------
TAR_PATH="${TAR_PATH:-/global/homes/s/sthapa/qtraj-nlo/input/out_pPb_8TeV/with131Knptraj/trajectories.tgz}"
LIST_DIR="${LIST_DIR:-$SCRATCH/qtraj_bundle_lists}"
MANIFEST="${MANIFEST:-$LIST_DIR/manifest.txt}"
NQTRAJ="${NQTRAJ:-20}"
OUTROOT="${OUTROOT:-$SCRATCH/qtraj_outputs}"

# Optional: pre-extracted tree (zero-tar path if set)
EXTRACT_ROOT="${EXTRACT_ROOT:-}"      # e.g. EXTRACT_ROOT=$SCRATCH/qtraj_files

LOCALIZE_TAR="${LOCALIZE_TAR:-1}"     # used only if EXTRACT_ROOT is empty
TASK_TIMEOUT="${TASK_TIMEOUT:-1200s}" # per-L cap
RETRIES="${RETRIES:-2}"
# ---------------------------------------------

# Mirror directory now uses the ARRAY MASTER id so ALL tasks of the array land together.
ARRAY_MASTER="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
COLLECT_DIR="${COLLECT_DIR:-$SLURM_SUBMIT_DIR/output_${ARRAY_MASTER}}"
# If a conflicting file exists, back it up once.
if [[ -e "$COLLECT_DIR" && ! -d "$COLLECT_DIR" ]]; then
  mv -f "$COLLECT_DIR" "${COLLECT_DIR}.bak.$(date +%s)"
fi
mkdir -p "$COLLECT_DIR"

: "${SLURM_ARRAY_TASK_ID:?Need an array index}"
PRANKS="$SLURM_NTASKS_PER_NODE"

BUNDLE_FILE="$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$MANIFEST")"
[[ -f "$BUNDLE_FILE" ]] || { echo "[FATAL] bundle not found: $BUNDLE_FILE"; exit 2; }

OUTDIR="$OUTROOT/${SLURM_JOB_ID}"   # each task keeps its own scratch jobid
mkdir -p "$OUTDIR"

BASENAME="$(basename "$BUNDLE_FILE")"
TAROUT="$OUTDIR/${BASENAME}.tar.gz"
DONEFLAG="$OUTDIR/${BASENAME}.done"

# Idempotent skip (and ensure mirrored copy exists too)
if [[ -f "$DONEFLAG" && -s "$TAROUT" ]]; then
  echo "[SKIP] $BASENAME already done."
  cp -f "$TAROUT" "$COLLECT_DIR/" || true
  exit 0
fi

# Basic env
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=TRUE OMP_PLACES=cores

NODE_TMP="${SLURM_TMPDIR:-/tmp}/qtraj.${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID}"
mkdir -p "$NODE_TMP/input/runset" "$NODE_TMP/chunks" "$NODE_TMP/run" "$NODE_TMP/failed"
cleanup(){ case "$NODE_TMP" in /|/tmp|/var/tmp|/dev/shm|"") echo "[WARN] refusing to rm $NODE_TMP";; *) rm -rf "$NODE_TMP" || true;; esac; }
trap cleanup EXIT

cp "$SLURM_SUBMIT_DIR/qtraj" "$NODE_TMP/qtraj"
chmod +x "$NODE_TMP/qtraj"
rsync -a --delete --exclude 'runset/' "$SLURM_SUBMIT_DIR/input/" "$NODE_TMP/input/"

WANTED=$(wc -l < "$BUNDLE_FILE")
START_TS=$(date +%s)

echo "[INFO] Host=$(hostname)"
echo "[INFO] Bundle=$BASENAME  Ranks=$PRANKS  NQTRAJ=$NQTRAJ  TIMEOUT=$TASK_TIMEOUT  RETRIES=$RETRIES"
echo "[INFO] OUTDIR=$OUTDIR"
echo "[INFO] COLLECT_DIR=$COLLECT_DIR"

# --- Bring inputs for this bundle to node-local ---
if [[ -n "$EXTRACT_ROOT" ]]; then
  echo "[INFO] Localizing from pre-extracted tree: $EXTRACT_ROOT"
  rsync -a --files-from="$BUNDLE_FILE" "$EXTRACT_ROOT/" "$NODE_TMP/input/runset/"
else
  TAR_SRC="$TAR_PATH"
  if (( LOCALIZE_TAR == 1 )); then
    TAR_LOCAL="$NODE_TMP/trajectories.tgz"
    echo "[INFO] Broadcasting tar to node-local via sbcast: $TAR_LOCAL"
    sbcast -f "$TAR_PATH" "$TAR_LOCAL"
    TAR_SRC="$TAR_LOCAL"
  fi
  echo "[INFO] Extracting bundle files from tar…"
  tar -xzf "$TAR_SRC" -C "$NODE_TMP/input/runset" --files-from "$BUNDLE_FILE"
fi

EXTRACTED=$(find "$NODE_TMP/input/runset" -type f | wc -l | tr -d ' ')
echo "[INFO] Extracted=$EXTRACTED  Wanted=$WANTED"
(( EXTRACTED == WANTED )) || echo "[WARN] extracted != wanted (check bundle paths)"

# --- Preflight (excluded from packaging) ---
PF="$NODE_TMP/preflight"; mkdir -p "$PF"; ln -sfn "$NODE_TMP/input" "$PF/input"
SAMPLE=$(head -n 1 "$BUNDLE_FILE")
echo "[INFO] Preflight sample: $SAMPLE"
( cd "$PF" && timeout 90s "$NODE_TMP/qtraj" -initL 0 -nTrajectories 1 \
    -temperatureEvolution 2 -temperatureFile "$NODE_TMP/input/runset/$SAMPLE" \
    -dirnameWithSeed 1 -outputSummaryFile 0 \
    >"preflight.out" 2>"preflight.err" ) || {
  echo "[FATAL] Preflight failed; tail follows:"
  tail -n +1 "$PF/preflight.err" | sed 's/^/[PREFLIGHT-ERR] /'
  exit 11
}

# --- Split into per-rank chunk lists (3-digit suffix) ---
split -n r/$PRANKS -d -a 3 --additional-suffix=.list \
      "$BUNDLE_FILE" "$NODE_TMP/chunks/chunk_"

# --- Worker script ---
WORKER="$NODE_TMP/worker.sh"
cat > "$WORKER" <<'EOS'
#!/bin/bash
set -uo pipefail
LOCAL_RANK=${SLURM_LOCALID:?}
BASE="__BASE__"
QTRAJ="$BASE/qtraj"
NQ="__NQ__"
TO="__TASK_TIMEOUT__"
RT="__RETRIES__"
LISTFILE="$BASE/chunks/chunk_$(printf "%03d" "$LOCAL_RANK").list"
mkdir -p "$BASE/run/$LOCAL_RANK" "$BASE/failed"

run_one () {
  local rel="$1" L="$2"
  local abs="$BASE/input/runset/$rel"
  local tag; tag=$(echo -n "$rel" | sha1sum | awk '{print $1}')
  local work="$BASE/run/$LOCAL_RANK/$tag"
  mkdir -p "$work"
  ln -sfn "$BASE/input" "$work/input"
  local try=0
  while (( try <= RT )); do
    if ( cd "$work" && timeout "$TO" "$QTRAJ" -initL "$L" -nTrajectories "$NQ" \
           -temperatureEvolution 2 -temperatureFile "$abs" \
           -dirnameWithSeed 1 -outputSummaryFile 0 \
           >"qtraj.L${L}.out" 2>"qtraj.L${L}.err" ); then
      return 0
    fi
    ((try++)); sleep 1
  done
  echo "$rel L=$L" >> "$BASE/failed/rank_${LOCAL_RANK}.txt"
  return 1
}

count=0
if [[ -s "$LISTFILE" ]]; then
  while IFS= read -r rel; do
    echo "[rank $LOCAL_RANK] file=$rel"
    run_one "$rel" 0 || true
    run_one "$rel" 1 || true
    ((count++))
  done < "$LISTFILE"
else
  echo "[rank $LOCAL_RANK] WARNING: empty $LISTFILE"
fi
echo "[rank $LOCAL_RANK] processed $count files"
exit 0
EOS
sed -i "s#__BASE__#$NODE_TMP#g"            "$WORKER"
sed -i "s#__NQ__#$NQTRAJ#g"                 "$WORKER"
sed -i "s#__TASK_TIMEOUT__#$TASK_TIMEOUT#g" "$WORKER"
sed -i "s#__RETRIES__#$RETRIES#g"           "$WORKER"
chmod +x "$WORKER"

echo "[INFO] Launching $PRANKS ranks…"
srun --kill-on-bad-exit=0 \
     --distribution=block:block \
     --ntasks="$PRANKS" \
     --cpus-per-task=1 \
     --cpu-bind=cores --mem-bind=local \
     "$WORKER" || true

# --- Package ONLY run/*/ratios.tsv (+ logs/failures). Exclude preflight. ---
echo "[INFO] Saving results to $TAROUT"
find "$NODE_TMP/run" -type f -name 'ratios.tsv' -print0 \
  -o -type f -name 'qtraj.L*.out' -print0 \
  -o -type f -name 'qtraj.L*.err' -print0 \
  -o -path "$NODE_TMP/failed/*" -type f -print0 \
  | tar --null -T - -czf "$TAROUT"
touch "$DONEFLAG"

# Mirror to submit dir (grouped by array master id)
cp -f "$TAROUT" "$COLLECT_DIR/"

OK=$(tar -tzf "$TAROUT" | grep -c '/ratios\.tsv$' || true)
BAD=$(tar -tzf "$TAROUT" | grep -c '^.*/failed/rank_.*\.txt$' || true)
ELAPSED=$(( $(date +%s) - START_TS ))
EXPECT=$(( 2 * WANTED ))
THRU=$(awk -v ok="$OK" -v e="$ELAPSED" 'BEGIN{if(e>0) printf "%.2f", ok/e; else print 0}')
echo "[SUMMARY] wanted=$WANTED expected=$EXPECT got=$OK  failures=$BAD  bundle=$BASENAME  elapsed=${ELAPSED}s  rate=${THRU} files/s"
echo "[DONE] $BASENAME  $(date)"

