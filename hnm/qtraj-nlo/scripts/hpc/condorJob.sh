#!/bin/bash 

if [ $# -eq 0 ]
then
    echo "Usage: run.sh <jobid>"
    exit
fi

# load job id from first argument
JOBID=$1

# number of phyical trajectories to sample
NPTRAJ=100

# number of quantum trajectories to sample per physical trajectory
NQTRAJ=1

for (( c=1; c<=NPTRAJ; c++ ))
do  
   f="$(mktemp /tmp/trajectory.XXXXXXXXXXXX)"
   curl http://45.79.164.44/ > ${f}
   ./qtraj -initN 1 -initL 0 -dirnameWithSeed 0 -nTrajectories ${NQTRAJ} -temperatureEvolution 2 -temperatureFile ${f}
   ./qtraj -initN 2 -initL 1 -dirnameWithSeed 0 -nTrajectories ${NQTRAJ} -temperatureEvolution 2 -temperatureFile ${f}
   gzip -c output/ratios.tsv >> datafile-${JOBID}.gz
   rm -rf output
done
