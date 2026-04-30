#!/bin/bash

#SBATCH -t 12:00:00             # Run time (hh:mm:ss)
#SBATCH -p qcd                  # Queue (partition) name
#SBATCH -J qtraj                # Job name
#SBATCH --nodes=1               # Number of nodes
#SBATCH --tasks-per-node=200    # Number of parallel jobs to launch
#SBATCH -o log.o%j              # Name of stdout output file
#SBATCH -e log.e%j              # Name of stderr error file

##################################################################################
echo "--- START OF SCRIPT ---"
date
echo "Current date and time: $(date)"

# Define working directories
JOBID=${SLURM_JOB_ID}
echo "Job ID: $JOBID"
jobDir=${SLURM_SUBMIT_DIR}
echo "Job Directory: $jobDir"

# Trajectory File
trajFile=$jobDir/input/runset/trajectories.tgz
echo "Trajectory File: $trajFile"

echo "Creating Output Directory"
outputDir="$jobDir/output-${JOBID}"
mkdir -p "$outputDir"
echo "Output Directory: $outputDir"

#################################################################################
NPTRAJ=1000
echo "Number of physical trajectories to sample (NPTRAJ): $NPTRAJ"
NQTRAJ=20
echo "Number of quantum trajectories per physical trajectory (NQTRAJ): $NQTRAJ"
##################################################################################

echo "Creating Temporary Directory for Execution of Job"
tempDir=$(mktemp -d -p /scratch "qtraj_$JOBID_XXXXXXX")
echo "Temporary Directory for execution: $tempDir"
cd "$tempDir"
echo "Current directory: $(pwd)"

# Setup local directory structure
mkdir -p ./input/runset
echo "Temp Directory with input and runset folders created"

# Copy necessary input files
echo "=====COPYING FILES INTO TEMP SPACE======"
copy_if_diff() {
    if [ "$(realpath "$1")" != "$(realpath "$2")" ]; then
        cp "$1" "$2"
    fi
}
copy_if_diff "$jobDir/input/params.txt" "./input/params.txt"
copy_if_diff "$jobDir/input/basisfunction*.tsv" "./input/"
copy_if_diff "$jobDir/qtraj" "./qtraj"
chmod +x ./qtraj  # Ensure qtraj is executable
echo "Copied input files to temporary directory"

##################################################################################
# Extract trajectories
cd ./input/runset
echo "Now in runset folder: $(pwd)"
tar -ztf "$trajFile" | shuf -n ${NPTRAJ} > trajectoryList.txt
echo "Extracting trajectories...."
tar --warning=no-timestamp -zxf "$trajFile" --files-from trajectoryList.txt
echo "Done extracting trajectories!"

# Return to temporary working space
cd "$tempDir"
echo "Back in temp directory: $(pwd)"

echo "=========QTRAJ EXECUTION STARTS HERE================="
cnt=0
for file in ./input/runset/*; do
    if [[ "$file" == *.txt && "$file" != ./input/runset/trajectoryList.txt ]]; then
        cnt=$((cnt+1))
        (
            mkdir -p "output_${cnt}_l0"
            cd "output_${cnt}_l0"
            ../qtraj -initL 0 -nTrajectories ${NQTRAJ} -temperatureEvolution 2 -temperatureFile "../../$file" > output.log 2>&1
        ) &
        (
            mkdir -p "output_${cnt}_l1"
            cd "output_${cnt}_l1"
            ../qtraj -initL 1 -nTrajectories ${NQTRAJ} -temperatureEvolution 2 -temperatureFile "../../$file" > output.log 2>&1
        ) &
        if (( cnt % 100 == 0 )); then
            wait  # Sync every 100 trajectories to limit concurrent jobs
        fi
    fi
done
wait
echo "Completed execution for all trajectory files"

#####################################################################################
echo "Aggregating and compressing results..."
# Aggregate and compress ratios.tsv files
find . -type f -name "ratios.tsv" -print0 | xargs -0 cat | gzip > datafile.gz || { echo "Error: Failed to aggregate ratios.tsv. Preserving temporary files in $tempDir"; exit 1; }
mv datafile.gz "$outputDir/datafile.gz" || { echo "Error: Failed to move datafile.gz. Preserving temporary files in $tempDir"; exit 1; }
echo "Results copied to $outputDir/datafile.gz"

# Aggregate log files (optional)
find . -type f -name "*.log" -print0 | xargs -0 cat > qtraj_results.log || { echo "Error: Failed to aggregate log files. Preserving temporary files in $tempDir"; exit 1; }
mv qtraj_results.log "$outputDir/qtraj_results.log" || { echo "Error: Failed to move qtraj_results.log. Preserving temporary files in $tempDir"; exit 1; }
echo "Log files copied to $outputDir/qtraj_results.log"

echo "====QTRAJ RUNS FOR $NPTRAJ AND DATA AGGREGATION COMPLETED==="

# Cleanup only if all previous steps succeeded
echo "Cleaning up temporary directory..."
find . -type f -name "ratios.tsv" -delete
find . -type f -name "*.log" -delete
rm -rf "$tempDir"
echo "Cleanup completed. Temporary directory removed: $tempDir"

date
echo "Current date and time at completion: $(date)"
echo "--------------- END OF SCRIPT --------------------"
