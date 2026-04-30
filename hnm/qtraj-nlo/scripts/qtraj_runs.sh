#!/bin/bash -l
start_time=$(date +%s)   # record start time
echo "--- START OF SCRIPT ---"
date

TRAJ_DIR="$(pwd)/input/OxOx5360_Traj"
RUNSET_DIR="$(pwd)/input/runset"
OUTPUT_DIR="$(pwd)/output"
mkdir -p "$OUTPUT_DIR"

NPTRAJ=1000
NQTRAJ=20

echo "Extracting $NPTRAJ physical trajectories with $NQTRAJ quantum trajectories each!"

# Extract NPTRAJ random files
tar -ztf "${TRAJ_DIR}/trajectories.tgz" | shuf -n "$NPTRAJ" > "${RUNSET_DIR}/trajectoryList.txt"
tar -zxf "${TRAJ_DIR}/trajectories.tgz" --directory "$RUNSET_DIR" --files-from "${RUNSET_DIR}/trajectoryList.txt"

./scripts/helpers/cleandatafiles.sh

echo "=== Running QTraj ==="

# Loop over extracted trajectory files
for file in $(cat "${RUNSET_DIR}/trajectoryList.txt"); do
    fullpath="${RUNSET_DIR}/${file}"
    
    ./qtraj -initL 0 -nTrajectories ${NQTRAJ} -temperatureEvolution 2 -temperatureFile "$fullpath" 2>&1
    ./qtraj -initL 1 -nTrajectories ${NQTRAJ} -temperatureEvolution 2 -temperatureFile "$fullpath" 2>&1
done

# ✅ Cleanup: delete everything except trajectories.tgz
find "${RUNSET_DIR}" -mindepth 1 ! -name 'trajectories.tgz' -exec rm -rf {} +


echo "--- QTRAJ DONE ---"
date

end_time=$(date +%s)   # record end time
elapsed=$(( end_time - start_time ))

# Convert elapsed seconds to HH:MM:SS
hours=$(( elapsed / 3600 ))
mins=$(( (elapsed % 3600) / 60 ))
secs=$(( elapsed % 60 ))

echo "Total runtime: ${hours}h ${mins}m ${secs}s"

echo "--- DONE ---"
