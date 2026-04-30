#!/bin/bash -l
# prep_bundles.slurm — create bundle lists & manifest from trajectories.tgz
# Usage:
#   sbatch scripts/prep_bundles.slurm \
#     --export=ALL,TAR_PATH=/path/to/trajectories.tgz,BUNDLE_SIZE=1024
#
# Author: Sabin Thapa <sthapa3@kent.edu>

#SBATCH -A m5098
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -t 00:20:00
#SBATCH -J prep_bundles
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
set -euo pipefail

# ---- User knobs ----
TAR_PATH="${TAR_PATH:-/global/homes/s/sthapa/qtraj-nlo/input/out_pPb_8TeV/with131Knptraj/trajectories.tgz}"
BUNDLE_SIZE="${BUNDLE_SIZE:-4096}"     # recommend: 4096 for production
SCRATCH_DIR="${SCRATCH_DIR:-$SCRATCH/qtraj_bundles}"
LIST_DIR="${LIST_DIR:-$SCRATCH/qtraj_bundle_lists}"
MANIFEST="${MANIFEST:-$LIST_DIR/manifest.txt}"
mkdir -p "$SCRATCH_DIR" "$LIST_DIR"
# --------------------

echo "[INFO] Host=$(hostname)"
echo "[INFO] Indexing $TAR_PATH (no extraction)…"

ALL_LIST="$SCRATCH_DIR/all_files.list"
rm -f "$ALL_LIST" "$LIST_DIR"/bundle_*.list "$MANIFEST" 2>/dev/null || true

tar -tzf "$TAR_PATH" | awk '!/\/$/{sub(/^\.\//,""); print}' > "$ALL_LIST"
TOTAL=$(wc -l < "$ALL_LIST")
echo "[INFO] NPTRAJs found: $TOTAL"

# Duplicate guard
if sort "$ALL_LIST" | uniq -d | head -n1 | grep -q . ; then
  echo "[FATAL] Duplicate NPTRAJ path detected:"
  sort "$ALL_LIST" | uniq -d | head -n5 | sed 's/^/  /'
  exit 3
fi

echo "[INFO] Creating bundles of $BUNDLE_SIZE files…"
split -d -l "$BUNDLE_SIZE" --additional-suffix=.list "$ALL_LIST" "$LIST_DIR/bundle_"
ls -1 "$LIST_DIR"/bundle_*.list | sort > "$MANIFEST"
NBUNDLES=$(wc -l < "$MANIFEST")
echo "[INFO] Bundles: $NBUNDLES"
head -n 5 "$MANIFEST" | sed 's/^/  /'

# Completeness check
SUM=$(xargs -a "$MANIFEST" -r cat | wc -l)
[[ "$SUM" -eq "$TOTAL" ]] || { echo "[FATAL] Sum($SUM) != Total($TOTAL)"; exit 4; }

echo "[DONE] prep_bundles  $(date)"
