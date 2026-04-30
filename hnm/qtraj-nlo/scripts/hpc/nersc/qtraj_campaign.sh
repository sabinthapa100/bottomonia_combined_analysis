#!/bin/bash -l
set -euo pipefail

MODE="${1:-}"
[[ -n "$MODE" ]] || {
  echo "Usage: $0 {prep|smoke|bench|prod|combine|status}"
  exit 2
}

# =========================
# User defaults
# =========================
ACCOUNT="${ACCOUNT:-m5098}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/qtraj-nlo}"
STATE_FILE="${PROJECT_ROOT}/.qtraj_last_runroot"

CAMPAIGN_TAG="${CAMPAIGN_TAG:-OO5360_mb_300k}"
INPUT_TAR="${INPUT_TAR:-$PROJECT_ROOT/input/out_OO5360_mb_300k/trajectories.tgz}"

BUNDLE_SIZE="${BUNDLE_SIZE:-1024}"
NQTRAJ="${NQTRAJ:-20}"

LOCALIZE_TAR="${LOCALIZE_TAR:-1}"
TASK_TIMEOUT="${TASK_TIMEOUT:-1800s}"
RETRIES="${RETRIES:-2}"
SAVE_TASK_LOGS="${SAVE_TASK_LOGS:-0}"

SMOKE_WALL="${SMOKE_WALL:-01:15:00}"
BENCH_WALL="${BENCH_WALL:-01:15:00}"
PROD_WALL="${PROD_WALL:-01:15:00}"

SMOKE_ARRAY="${SMOKE_ARRAY:-0-0}"
BENCH_ARRAY="${BENCH_ARRAY:-0-3%2}"
PROD_THROTTLE="${PROD_THROTTLE:-16}"

# =========================
# Helpers
# =========================
load_env() {
  module load cpu
  module load PrgEnv-gnu
  module load cray-fftw/3.3.10.11
}

build_qtraj() {
  load_env
  cd "$PROJECT_ROOT"
  make clean
  make -j 16
  [[ -x "$PROJECT_ROOT/qtraj" ]] || {
    echo "[FATAL] qtraj build failed"
    exit 10
  }
}

new_runroot() {
  RUNROOT="${SCRATCH}/qtraj_runs/${CAMPAIGN_TAG}_$(date +%Y%m%d_%H%M%S)"
  echo "$RUNROOT" > "$STATE_FILE"
}

use_existing_runroot() {
  if [[ -n "${RUNROOT:-}" ]]; then
    :
  elif [[ -f "$STATE_FILE" ]]; then
    RUNROOT="$(cat "$STATE_FILE")"
  else
    echo "[FATAL] No existing campaign. Run: $0 prep"
    exit 11
  fi
}

set_paths() {
  RUNTIME="${RUNROOT}/runtime"
  INPUTDIR="${RUNROOT}/input"
  LISTDIR="${RUNROOT}/lists"
  BUNDLEDIR="${RUNROOT}/bundles"
  OUTROOT="${RUNROOT}/outputs"
  COLLECTDIR="${RUNROOT}/collect"
  FINALDIR="${RUNROOT}/final"
  LOGDIR="${RUNROOT}/logs"
  METADIR="${RUNROOT}/meta"
  TAR_PATH="${INPUTDIR}/trajectories.tgz"
  MANIFEST="${LISTDIR}/manifest.txt"
}

print_info() {
  echo "[INFO] RUNROOT=$RUNROOT"
  echo "[INFO] TAR_PATH=$TAR_PATH"
  echo "[INFO] MANIFEST=$MANIFEST"
  echo "[INFO] BUNDLE_SIZE=$BUNDLE_SIZE"
  echo "[INFO] NQTRAJ=$NQTRAJ"
  echo "[INFO] OUTROOT=$OUTROOT"
  echo "[INFO] COLLECTDIR=$COLLECTDIR"
  echo "[INFO] FINALDIR=$FINALDIR"
}

stage_campaign() {
  mkdir -p "$RUNROOT"/{runtime,input,lists,bundles,outputs,collect,final,logs,meta}

  # copy repo skeleton to runtime
  rsync -a "$PROJECT_ROOT/" "$RUNTIME/" \
    --exclude '.git' \
    --exclude 'output' \
    --exclude 'build' \
    --exclude '.qtraj_last_runroot'

  # copy lightweight input tree, exclude heavy tar
  mkdir -p "$RUNTIME/input"
  rsync -a "$PROJECT_ROOT/input/" "$RUNTIME/input/" \
    --exclude 'out_OO5360_mb_300k/trajectories.tgz'

  # stage production tar to scratch only once
  cp "$INPUT_TAR" "$TAR_PATH"

  # stage executable into runtime
  cp "$PROJECT_ROOT/qtraj" "$RUNTIME/qtraj"
  chmod +x "$RUNTIME/qtraj"

  echo "$RUNROOT"  > "$METADIR/runroot.txt"
  echo "$TAR_PATH" > "$METADIR/tar_path.txt"
}

write_helper_scripts() {
  mkdir -p "$RUNTIME/scripts"

  cat > "$RUNTIME/scripts/prep_bundles.slurm" <<'EOF_PREP'
#!/bin/bash -l
#SBATCH -A m5098
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -t 00:20:00
#SBATCH -J prep_bundles
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail

: "${TAR_PATH:?Need TAR_PATH}"
: "${BUNDLE_SIZE:?Need BUNDLE_SIZE}"
: "${SCRATCH_DIR:?Need SCRATCH_DIR}"
: "${LIST_DIR:?Need LIST_DIR}"
: "${MANIFEST:?Need MANIFEST}"

mkdir -p "$SCRATCH_DIR" "$LIST_DIR"

echo "[INFO] Host=$(hostname)"
echo "[INFO] Indexing $TAR_PATH (no extraction)..."

ALL_LIST="$SCRATCH_DIR/all_files.list"
rm -f "$ALL_LIST" "$LIST_DIR"/bundle_*.list "$MANIFEST" 2>/dev/null || true

tar -tzf "$TAR_PATH" | awk '!/\/$/{sub(/^\.\//,""); print}' > "$ALL_LIST"
TOTAL=$(wc -l < "$ALL_LIST")
echo "[INFO] NPTRAJs found: $TOTAL"

if sort "$ALL_LIST" | uniq -d | head -n1 | grep -q . ; then
  echo "[FATAL] Duplicate NPTRAJ path detected:"
  sort "$ALL_LIST" | uniq -d | head -n5 | sed 's/^/  /'
  exit 3
fi

echo "[INFO] Creating bundles of $BUNDLE_SIZE files..."
split -d -l "$BUNDLE_SIZE" --additional-suffix=.list "$ALL_LIST" "$LIST_DIR/bundle_"
ls -1 "$LIST_DIR"/bundle_*.list | sort > "$MANIFEST"

NBUNDLES=$(wc -l < "$MANIFEST")
echo "[INFO] Bundles: $NBUNDLES"
head -n 5 "$MANIFEST" | sed 's/^/  /'

SUM=$(xargs -a "$MANIFEST" -r cat | wc -l)
[[ "$SUM" -eq "$TOTAL" ]] || { echo "[FATAL] Sum($SUM) != Total($TOTAL)"; exit 4; }

echo "[DONE] prep_bundles $(date)"
EOF_PREP

  cat > "$RUNTIME/scripts/qtraj_node.slurm" <<'EOF_NODE'
#!/bin/bash -l
#SBATCH -A m5098
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1 --exclusive
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --signal=B:USR1@120

set -euo pipefail
umask 002

: "${RUNTIME_ROOT:?Need RUNTIME_ROOT}"
: "${TAR_PATH:?Need TAR_PATH}"
: "${MANIFEST:?Need MANIFEST}"
: "${OUTROOT:?Need OUTROOT}"
: "${COLLECT_DIR:?Need COLLECT_DIR}"

NQTRAJ="${NQTRAJ:-20}"
LOCALIZE_TAR="${LOCALIZE_TAR:-1}"
TASK_TIMEOUT="${TASK_TIMEOUT:-1800s}"
RETRIES="${RETRIES:-2}"
SAVE_TASK_LOGS="${SAVE_TASK_LOGS:-0}"

: "${SLURM_ARRAY_TASK_ID:?Need array index}"

PRANKS="${SLURM_NTASKS_PER_NODE:-128}"
BUNDLE_FILE="$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$MANIFEST")"
[[ -f "$BUNDLE_FILE" ]] || { echo "[FATAL] bundle not found: $BUNDLE_FILE"; exit 2; }

BASENAME="$(basename "$BUNDLE_FILE")"        # bundle_00.list
BUNDLE_KEY="${BASENAME%.list}"               # bundle_00

STATE_ROOT="${OUTROOT}/_state"
BUNDLE_STATE="${STATE_ROOT}/${BUNDLE_KEY}"
RESULTS_DIR="${BUNDLE_STATE}/results"
FAILED_DIR="${BUNDLE_STATE}/failed"
TASKLOG_DIR="${BUNDLE_STATE}/tasklogs"
STATUS_FILE="${BUNDLE_STATE}/status.txt"
DONEFLAG="${BUNDLE_STATE}/${BASENAME}.done"
TAROUT="${BUNDLE_STATE}/${BASENAME}.tar.gz"

mkdir -p "$RESULTS_DIR" "$FAILED_DIR" "$TASKLOG_DIR" "$COLLECT_DIR"

if [[ -f "$DONEFLAG" && -s "$TAROUT" ]]; then
  cp -f "$TAROUT" "$COLLECT_DIR/" || true
  echo "[SKIP] $BASENAME already complete"
  exit 0
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores

NODE_TMP="${SLURM_TMPDIR:-/tmp}/qtraj.${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID}"
STOPFLAG="${NODE_TMP}/STOP_REQUESTED"
mkdir -p "$NODE_TMP/input/runset" "$NODE_TMP/chunks" "$NODE_TMP/run"

cleanup() {
  case "$NODE_TMP" in
    /|/tmp|/var/tmp|/dev/shm|"")
      echo "[WARN] refusing to rm $NODE_TMP"
      ;;
    *)
      rm -rf "$NODE_TMP" || true
      ;;
  esac
}
trap cleanup EXIT

on_usr1() {
  echo "[WARN] Batch shell got USR1, requesting graceful stop"
  touch "$STOPFLAG"
}
trap on_usr1 USR1

cp "$RUNTIME_ROOT/qtraj" "$NODE_TMP/qtraj"
chmod +x "$NODE_TMP/qtraj"
rsync -a --delete --exclude 'runset/' "$RUNTIME_ROOT/input/" "$NODE_TMP/input/"

WANTED=$(wc -l < "$BUNDLE_FILE")
EXPECT=$((2 * WANTED))
START_TS=$(date +%s)

echo "[INFO] Host=$(hostname)"
echo "[INFO] Bundle=$BASENAME Ranks=$PRANKS NQTRAJ=$NQTRAJ TIMEOUT=$TASK_TIMEOUT RETRIES=$RETRIES"
echo "[INFO] STATE=$BUNDLE_STATE"
echo "[INFO] COLLECT_DIR=$COLLECT_DIR"

if [[ -n "${EXTRACT_ROOT:-}" ]]; then
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
  echo "[INFO] Extracting bundle files from tar..."
  tar -xzf "$TAR_SRC" -C "$NODE_TMP/input/runset" --files-from "$BUNDLE_FILE"
fi

EXTRACTED=$(find "$NODE_TMP/input/runset" -type f | wc -l | tr -d ' ')
echo "[INFO] Extracted=$EXTRACTED Wanted=$WANTED"

PF="$NODE_TMP/preflight"
mkdir -p "$PF"
ln -sfn "$NODE_TMP/input" "$PF/input"

SAMPLE=$(head -n 1 "$BUNDLE_FILE")
echo "[INFO] Preflight sample: $SAMPLE"

(
  cd "$PF"
  timeout 90s "$NODE_TMP/qtraj" \
    -initL 0 \
    -nTrajectories 1 \
    -temperatureEvolution 2 \
    -temperatureFile "$NODE_TMP/input/runset/$SAMPLE" \
    -dirnameWithSeed 1 \
    -outputSummaryFile 0 \
    > preflight.out 2> preflight.err
) || {
  echo "[FATAL] Preflight failed"
  tail -n +1 "$PF/preflight.err" | sed 's/^/[PREFLIGHT-ERR] /'
  exit 11
}

split -n r/$PRANKS -d -a 3 --additional-suffix=.list \
  "$BUNDLE_FILE" "$NODE_TMP/chunks/chunk_"

WORKER="$NODE_TMP/worker.sh"
cat > "$WORKER" <<EOF_WORKER
#!/bin/bash
set -uo pipefail

LOCAL_RANK=\${SLURM_LOCALID:?}
BASE="$NODE_TMP"
QTRAJ="\$BASE/qtraj"
NQ="$NQTRAJ"
TO="$TASK_TIMEOUT"
RT="$RETRIES"
RESULTS_DIR="$RESULTS_DIR"
FAILED_DIR="$FAILED_DIR"
TASKLOG_DIR="$TASKLOG_DIR"
SAVE_TASK_LOGS="$SAVE_TASK_LOGS"
STOPFLAG="$STOPFLAG"
LISTFILE="\$BASE/chunks/chunk_\$(printf "%03d" "\$LOCAL_RANK").list"

mkdir -p "\$RESULTS_DIR" "\$FAILED_DIR" "\$TASKLOG_DIR"

run_one() {
  local rel="\$1"
  local L="\$2"
  local tag
  tag=\$(echo -n "\$rel" | sha1sum | awk '{print \$1}')

  local target="\$RESULTS_DIR/\${tag}.L\${L}.ratios.tsv"
  local failfile="\$FAILED_DIR/rank_\${LOCAL_RANK}.txt"

  [[ -s "\$target" ]] && return 0
  [[ -e "\$STOPFLAG" ]] && return 99

  local abs="\$BASE/input/runset/\$rel"
  local work="\$BASE/run/\$LOCAL_RANK/\${tag}.L\${L}"
  local try=0

  rm -rf "\$work"
  mkdir -p "\$work"
  ln -sfn "\$BASE/input" "\$work/input"

  while (( try <= RT )); do
    rm -rf "\$work"/output*
    if (
      cd "\$work" && timeout "\$TO" "\$QTRAJ" \
        -initL "\$L" \
        -nTrajectories "\$NQ" \
        -temperatureEvolution 2 \
        -temperatureFile "\$abs" \
        -dirnameWithSeed 1 \
        -outputSummaryFile 0 \
        > "qtraj.L\${L}.out" 2> "qtraj.L\${L}.err"
    ); then
      local ratios_path
      ratios_path=\$(find "\$work" -type f -name 'ratios.tsv' | head -n 1 || true)
      if [[ -n "\$ratios_path" && -s "\$ratios_path" ]]; then
        cp "\$ratios_path" "\${target}.tmp.\$\$"
        mv -f "\${target}.tmp.\$\$" "\$target"

        if (( SAVE_TASK_LOGS == 1 )); then
          cp -f "\$work/qtraj.L\${L}.out" "\$TASKLOG_DIR/\${tag}.L\${L}.out" || true
          cp -f "\$work/qtraj.L\${L}.err" "\$TASKLOG_DIR/\${tag}.L\${L}.err" || true
        fi
        return 0
      fi
    fi
    ((try++))
    sleep 1
  done

  echo "\$rel L=\$L" >> "\$failfile"
  return 1
}

count=0
if [[ -s "\$LISTFILE" ]]; then
  while IFS= read -r rel; do
    [[ -z "\$rel" ]] && continue
    [[ -e "\$STOPFLAG" ]] && break

    echo "[rank \$LOCAL_RANK] file=\$rel"

    run_one "\$rel" 0
    rc=\$?
    (( rc == 99 )) && break

    [[ -e "\$STOPFLAG" ]] && break

    run_one "\$rel" 1
    rc=\$?
    (( rc == 99 )) && break

    ((count++))
  done < "\$LISTFILE"
else
  echo "[rank \$LOCAL_RANK] WARNING: empty \$LISTFILE"
fi

echo "[rank \$LOCAL_RANK] processed \$count files"
exit 0
EOF_WORKER
chmod +x "$WORKER"

echo "[INFO] Launching $PRANKS ranks..."
srun --kill-on-bad-exit=0 \
     --distribution=block:block \
     --ntasks="$PRANKS" \
     --cpus-per-task=1 \
     --cpu-bind=cores \
     --mem-bind=local \
     "$WORKER" || true

OK=$(find "$RESULTS_DIR" -maxdepth 1 -type f -name '*.ratios.tsv' | wc -l | tr -d ' ')
BAD=$(find "$FAILED_DIR" -maxdepth 1 -type f | wc -l | tr -d ' ')
ELAPSED=$(( $(date +%s) - START_TS ))

{
  echo "bundle=$BASENAME"
  echo "wanted=$WANTED"
  echo "expected=$EXPECT"
  echo "got=$OK"
  echo "failfiles=$BAD"
  echo "elapsed_seconds=$ELAPSED"
  echo "timestamp=$(date)"
} > "$STATUS_FILE"

if [[ "$OK" -eq "$EXPECT" ]]; then
  echo "[INFO] Bundle complete. Packaging $TAROUT"
  tar -C "$BUNDLE_STATE" -czf "${TAROUT}.tmp.$$" results failed status.txt
  mv -f "${TAROUT}.tmp.$$" "$TAROUT"
  touch "$DONEFLAG"

  cp -f "$TAROUT" "${COLLECT_DIR}/.${BASENAME}.tmp.$$"
  mv -f "${COLLECT_DIR}/.${BASENAME}.tmp.$$" "${COLLECT_DIR}/${BASENAME}.tar.gz"
else
  echo "[WARN] Bundle incomplete: got=$OK expected=$EXPECT"
  echo "[WARN] Safe to rerun the same array later. Completed files are already preserved."
  exit 3
fi

THRU=$(awk -v ok="$OK" -v e="$ELAPSED" 'BEGIN{if(e>0) printf "%.3f", ok/e; else print 0}')
echo "[SUMMARY] wanted=$WANTED expected=$EXPECT got=$OK failures=$BAD bundle=$BASENAME elapsed=${ELAPSED}s rate=${THRU} ratios_per_s"
echo "[DONE] $BASENAME $(date)"
EOF_NODE

  cat > "$RUNTIME/scripts/combine_runs.sh" <<'EOF_COMB'
#!/usr/bin/env bash
set -euo pipefail

for kv in "$@"; do
  case "$kv" in
    *=*)
      key="${kv%%=*}"
      val="${kv#*=}"
      export "$key"="$val"
      ;;
  esac
done

OUTDIR="${OUTDIR:-}"
OUTFILE="${OUTFILE:-}"
MANIFEST="${MANIFEST:-}"

[[ -n "$OUTDIR" ]] || { echo "[FATAL] Provide OUTDIR=<path>"; exit 2; }
OUTDIR="$(readlink -f "$OUTDIR")"
[[ -d "$OUTDIR" ]] || { echo "[FATAL] OUTDIR not found: $OUTDIR"; exit 2; }
[[ -z "$OUTFILE" ]] && OUTFILE="$OUTDIR/datafile.gz"

mapfile -t TARS < <(find "$OUTDIR" -maxdepth 1 -type f -name 'bundle_*.list.tar.gz' | sort)
[[ ${#TARS[@]} -gt 0 ]] || { echo "[FATAL] No bundle tarballs in $OUTDIR"; exit 3; }

if [[ -n "$MANIFEST" ]]; then
  [[ -f "$MANIFEST" ]] || { echo "[FATAL] MANIFEST not found: $MANIFEST"; exit 4; }
  mapfile -t EXPECTED < <(awk -F/ '{print $NF}' "$MANIFEST" | sed 's/$/.tar.gz/')
  mapfile -t PRESENT  < <(printf "%s\n" "${TARS[@]}" | awk -F/ '{print $NF}')
  MISSING=$(comm -23 <(printf "%s\n" "${EXPECTED[@]}" | sort) <(printf "%s\n" "${PRESENT[@]}" | sort) || true)
  [[ -z "$MISSING" ]] || {
    echo "[FATAL] Missing bundle tar(s):"
    printf '  %s\n' $(printf "%s\n" "$MISSING" | head -n 20)
    exit 5
  }
fi

mkdir -p "$(dirname "$OUTFILE")"
tmp="${OUTFILE}.tmp"

for T in "${TARS[@]}"; do
  tar -xOzf "$T" --wildcards '*.ratios.tsv' || true
done | gzip -9 > "$tmp"

mv -f "$tmp" "$OUTFILE"
echo "[DONE] Wrote: $OUTFILE"

echo "Preview:"
zcat "$OUTFILE" | head -n 8 || true
LINES=$(zcat "$OUTFILE" | wc -l | tr -d ' ')
echo "Total lines: $LINES"
EOF_COMB

  chmod +x "$RUNTIME/scripts/combine_runs.sh"
}

submit_prep() {
  sbatch -A "$ACCOUNT" -L scratch -C cpu -q shared -t 00:20:00 \
    --chdir="$RUNTIME" \
    -o "$LOGDIR/prep.%j.out" \
    -e "$LOGDIR/prep.%j.err" \
    --export=ALL,TAR_PATH="$TAR_PATH",BUNDLE_SIZE="$BUNDLE_SIZE",SCRATCH_DIR="$BUNDLEDIR",LIST_DIR="$LISTDIR",MANIFEST="$MANIFEST" \
    "$RUNTIME/scripts/prep_bundles.slurm"
}

submit_array() {
  local tag="$1"
  local array_spec="$2"
  local wall="$3"

  sbatch -A "$ACCOUNT" -L scratch -C cpu -q regular -t "$wall" \
    --chdir="$RUNTIME" \
    -o "$LOGDIR/${tag}.%A_%a.out" \
    -e "$LOGDIR/${tag}.%A_%a.err" \
    --array="$array_spec" \
    --export=ALL,RUNTIME_ROOT="$RUNTIME",TAR_PATH="$TAR_PATH",MANIFEST="$MANIFEST",NQTRAJ="$NQTRAJ",LOCALIZE_TAR="$LOCALIZE_TAR",TASK_TIMEOUT="$TASK_TIMEOUT",RETRIES="$RETRIES",OUTROOT="$OUTROOT",COLLECT_DIR="$COLLECTDIR",SAVE_TASK_LOGS="$SAVE_TASK_LOGS" \
    "$RUNTIME/scripts/qtraj_node.slurm"
}

submit_prod() {
  [[ -f "$MANIFEST" ]] || { echo "[FATAL] Missing manifest: $MANIFEST"; exit 20; }
  NBUNDLES=$(wc -l < "$MANIFEST")
  [[ "$NBUNDLES" -gt 0 ]] || { echo "[FATAL] Empty manifest"; exit 21; }

  submit_array "prod" "0-$((NBUNDLES-1))%${PROD_THROTTLE}" "$PROD_WALL"
}

show_status() {
  echo "[INFO] RUNROOT=$RUNROOT"
  if [[ -f "$MANIFEST" ]]; then
    EXPECTED=$(wc -l < "$MANIFEST")
  else
    EXPECTED=0
  fi
  PRESENT=$(find "$COLLECTDIR" -maxdepth 1 -type f -name 'bundle_*.list.tar.gz' | wc -l | tr -d ' ')
  DONECNT=$(find "$OUTROOT/_state" -type f -name '*.done' 2>/dev/null | wc -l | tr -d ' ')
  echo "[INFO] bundles_expected=$EXPECTED"
  echo "[INFO] bundles_done=$DONECNT"
  echo "[INFO] bundle_tars_in_collect=$PRESENT"
  [[ -f "$FINALDIR/datafile.gz" ]] && ls -lh "$FINALDIR/datafile.gz"
}

run_combine() {
  mkdir -p "$FINALDIR"
  "$RUNTIME/scripts/combine_runs.sh" \
    OUTDIR="$COLLECTDIR" \
    OUTFILE="$FINALDIR/datafile.gz" \
    MANIFEST="$MANIFEST"
}

# =========================
# Main
# =========================
case "$MODE" in
  prep)
    new_runroot
    set_paths
    build_qtraj
    stage_campaign
    write_helper_scripts
    print_info
    submit_prep
    ;;
  smoke)
    use_existing_runroot
    set_paths
    print_info
    [[ -f "$MANIFEST" ]] || { echo "[FATAL] Manifest not ready yet. Wait for prep to finish."; exit 30; }
    submit_array "smoke" "$SMOKE_ARRAY" "$SMOKE_WALL"
    ;;
  bench)
    use_existing_runroot
    set_paths
    print_info
    [[ -f "$MANIFEST" ]] || { echo "[FATAL] Manifest not ready yet. Wait for prep to finish."; exit 31; }
    submit_array "bench" "$BENCH_ARRAY" "$BENCH_WALL"
    ;;
  prod)
    use_existing_runroot
    set_paths
    print_info
    [[ -f "$MANIFEST" ]] || { echo "[FATAL] Manifest not ready yet. Wait for prep to finish."; exit 32; }
    submit_prod
    ;;
  combine)
    use_existing_runroot
    set_paths
    print_info
    run_combine
    ;;
  status)
    use_existing_runroot
    set_paths
    show_status
    ;;
  *)
    echo "Usage: $0 {prep|smoke|bench|prod|combine|status}"
    exit 2
    ;;
esac
