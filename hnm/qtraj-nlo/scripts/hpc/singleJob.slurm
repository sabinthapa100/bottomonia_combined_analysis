#!/bin/bash

#SBATCH -J qtraj           # Job name
#SBATCH -o log.o%j         # Name of stdout output file
#SBATCH -e log.e%j         # Name of stderr error file
#SBATCH -p small           # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 00:30:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=none   # Do not send email at begin and end of job
#SBATCH -A PHY22035        # Project/Allocation name (req'd if you have more than 1)

date
./qtraj
date
