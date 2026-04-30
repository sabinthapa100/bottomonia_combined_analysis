#!/usr/bin/env bash
# combine_runs.sh — Concatenate all run/*/ratios.tsv from mirrored bundle tars.
# Usage (either works):
#   ./scripts/combine_runs.sh OUTDIR=$PWD/output_<JOBID> [OUTFILE=.../datafile.gz] [MANIFEST=.../manifest.txt]
#   OUTDIR=$PWD/output_<JOBID> MANIFEST=$SCRATCH/qtraj_bundle_lists/manifest.txt ./scripts/combine_runs.sh
#
# OUTDIR   = directory containing bundle_*.list.tar.gz (made by qtraj_runs.slurm mirror)
# OUTFILE  = final gzip, default: $OUTDIR/datafile.gz
# MANIFEST = optional manifest to check no bundles are missing
#
# Author: Sabin Thapa <sthapa3@kent.edu>

set -euo pipefail

# --- Parse KEY=VALUE args ---
for kv in "$@"; do
  case "$kv" in
    *=*)
      key="${kv%%=*}"
      val="${kv#*=}"
      export "$key"="$val"
      ;;
    *)
      echo "[WARN] Ignoring positional arg: $kv"
      ;;
  esac
done

OUTDIR="${OUTDIR:-}"
OUTFILE="${OUTFILE:-}"
MANIFEST="${MANIFEST:-}"

if [[ -z "$OUTDIR" ]]; then
  echo "[FATAL] Provide OUTDIR=<path> (the output_<JOBID> mirror dir)."
  exit 2
fi

OUTDIR="$(readlink -f "$OUTDIR")"
[[ -d "$OUTDIR" ]] || { echo "[FATAL] OUTDIR not found: $OUTDIR"; exit 2; }
[[ -z "$OUTFILE" ]] && OUTFILE="$OUTDIR/datafile.gz"

echo "[INFO] OUTDIR=$OUTDIR"
echo "[INFO] OUTFILE=$OUTFILE"
[[ -n "$MANIFEST" ]] && echo "[INFO] MANIFEST=$MANIFEST"

# --- Discover tars (one per bundle) ---
mapfile -t TARS < <(find "$OUTDIR" -maxdepth 1 -type f -name 'bundle_*.list.tar.gz' | sort)
NB_PRESENT=${#TARS[@]}
[[ $NB_PRESENT -gt 0 ]] || { echo "[FATAL] No bundle_*.list.tar.gz files in $OUTDIR"; exit 3; }
echo "[INFO] Found $NB_PRESENT bundle tarballs (showing up to 8):"
printf '  %s\n' "${TARS[@]:0:8}"; [[ $NB_PRESENT -gt 8 ]] && echo "  ..."

# --- Optional: verify against manifest (ensures no missing bundles) ---
if [[ -n "$MANIFEST" ]]; then
  [[ -f "$MANIFEST" ]] || { echo "[FATAL] MANIFEST not found: $MANIFEST"; exit 4; }
  NB_EXPECTED=$(wc -l < "$MANIFEST")
  echo "[INFO] Bundles expected from manifest: $NB_EXPECTED"

  # Expected basenames from manifest (bundle_xx.list -> bundle_xx.list.tar.gz)
  mapfile -t EXPECTED < <(awk -F/ '{print $NF}' "$MANIFEST" | sed 's/$/.tar.gz/')
  # Present basenames
  mapfile -t PRESENT  < <(printf "%s\n" "${TARS[@]}" | awk -F/ '{print $NF}')

  # Which expected are missing?
  MISSING=$(comm -23 <(printf "%s\n" "${EXPECTED[@]}" | sort) <(printf "%s\n" "${PRESENT[@]}" | sort) || true)
  if [[ -n "$MISSING" ]]; then
    echo "[FATAL] Missing bundle tar(s) (first 10 shown):"
    printf '  %s\n' $(printf "%s\n" "$MISSING" | head -n 10)
    exit 5
  fi
fi

# --- Combine: stream only run/*/ratios.tsv from each tar ---
tmp="${OUTFILE}.tmp"
mkdir -p "$(dirname "$OUTFILE")"
: > "$tmp"

echo "[INFO] Combining ratios.tsv from tarballs…"
# Loop avoids long argv lists and keeps memory modest.
for T in "${TARS[@]}"; do
  # paths inside the tar look like: .../run/<rank>/<hash>/ratios.tsv
  tar -xOzf "$T" --wildcards --no-anchored 'run/*/*/ratios.tsv' || true
done | gzip -9 > "$tmp"

mv -f "$tmp" "$OUTFILE"
echo "[DONE] Wrote: $OUTFILE"

# --- Quick stats ---
echo "Preview (first 8 lines):"
zcat "$OUTFILE" | head -n 8 || true

LINES=$(zcat "$OUTFILE" | wc -l | tr -d ' ')
echo "Total lines: $LINES"

