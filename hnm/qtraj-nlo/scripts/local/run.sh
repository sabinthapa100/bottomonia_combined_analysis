#!/usr/bin/env bash
# run.sh - Runs qtraj for NPTRAJ physical trajectories

# Setup
echo "Starting job.."
date
HOME_DIR=$(pwd)
RUNSET_DIR="$HOME_DIR/input/runset"
TRAJ_FILE="$HOME_DIR/input/OxOx5360_Traj/trajectories.tgz"
LIST_FILE="$RUNSET_DIR/trajectoryList.txt"
RUN_ID=$(date +%Y%m%d%H%M%S)
OUTPUT_DIR="$HOME_DIR/output_$RUN_ID"

NPTRAJ=10  # Physical trajectories
NQTRAJ=20   # Quantum trajectories per physical trajectory

# Create output directory
mkdir -p "$OUTPUT_DIR/snapshots"

# Extract trajectories
echo "Extracting $NPTRAJ trajectories..."
mkdir -p "$RUNSET_DIR"
tar -ztf "$TRAJ_FILE" | shuf -n $NPTRAJ > "$LIST_FILE"
tar -zxf "$TRAJ_FILE" -C "$RUNSET_DIR" --files-from "$LIST_FILE"
rm "$LIST_FILE"
echo "Extraction complete!"

# Run qtraj jobs **one at a time**
for traj in "$RUNSET_DIR"/trajectory_*.txt; do
  echo "Processing $traj"
  # L=0
  ./qtraj \
    -initL 0 \
    -nTrajectories "$NQTRAJ" \
    -temperatureEvolution 2 \
    -temperatureFile "$traj" \
    -dirnameWithSeed 1 \
    -outputSummaryFile 0 \
    > /dev/null 2>&1

  # L=1
  ./qtraj \
    -initL 1 \
    -nTrajectories "$NQTRAJ" \
    -temperatureEvolution 2 \
    -temperatureFile "$traj" \
    -dirnameWithSeed 1 \
    -outputSummaryFile 0 \
    > /dev/null 2>&1

  echo "Finished both L=0 and L=1 for $traj"
done

echo "All qtraj runs completed!"
date

# Collect results
echo "Collecting results..."
find "$HOME_DIR"/output-* -name "ratios.tsv" -exec cat {} + | gzip > "$OUTPUT_DIR/datafile.gz"

# Collect snapshots into common folder with subdirectories
for dir in "$HOME_DIR"/output-*; do
  [ -d "$dir" ] || continue
  seed=$(basename "$dir" | sed 's/output-//')
  mkdir -p "$OUTPUT_DIR/snapshots/$seed"
  cp "$dir"/snapshot_*.tsv "$OUTPUT_DIR/snapshots/$seed/" 2>/dev/null || true
done

# Cleanup
echo "Cleaning up..."
rm -rf "$HOME_DIR"/output-*
rm -rf "$RUNSET_DIR"/trajectory_*.txt

echo "Done. Outputs in: $OUTPUT_DIR"
date
exit 0
