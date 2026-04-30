============================================================
 qtraj-nlo
 Version 2.1
 Copyright (c) Michael Strickland
 Nov 6 2022
============================================================

------------------------------------------------------------
 DOCUMENTATION
------------------------------------------------------------

Documentation for the code and underlying physics can be 
found in the "docs" folder.

------------------------------------------------------------
 COMPILING
------------------------------------------------------------

To compile, simply type 

   make 

from the main code directory.

For full functionality, the following need to be installed 
on your sytem prior to compilation

  - FFTW 3.3.8+
  - GSL 2.6+
  - Armadillo 10.1.2+

In order to run the automatic unit tests

  - GoogleTest 1.10.0+

is required.

To run the unit tests, simply type

  make tests

in the main code directory.

------------------------------------------------------------
 USAGE
------------------------------------------------------------
FOR ONE PHYSICAL TRAJECTORY
------------------------------------------------------------
There is a file "input/params.txt" that contains all 
parameters that can be adjusted at runtime.  It includes 
comments describing the various options.  To run with the 
params.txt parameters simply type

  ./qtraj

If you would like to override some parameters in the 
params.txt file from the commandline the syntax is

  ./qtraj -<PARAMNAME1> <value1> ... -<PARAMNAMEn> <valuen>

------------------------------------------------------------
 OUTPUT
------------------------------------------------------------

If the 'dirnameWithSeed' parameter is set to 1, then all 
files are output into "output-<seed>"; otherwise, they are 
output into "output".  The latter is useful for testing.  
In both cases, if the directory doesn't exist, it is 
created.  It is possible to turn on/off the output of 
the time-evolved wavefunctions and/or summary files.  The
default is to output only to "output/ratios.tsv".  If 
the file "output/ratios.tsv" already exists, then the
results are appended to this file.

-------------------------------------------------------------
For Many Physical Trajectories
-------------------------------------------------------------
#####################################################################
In the ./input/runset/ directory
--------------------------------------------------------------
The script file to run qtraj for many physical trajectories 
is (manyTraj_local.sh is located in ./scripts/ folder)
--------------------------------------------------------------
### EXTRACTING Desired number of trajectories
tar -ztf trajectories.tgz | shuf -n 1000 > trajectoryList.txt
tar -zxf trajectories.tgz --files-from trajectoryList.txt
#####################################################################








------------------------------------------------------------
 LICENSE
------------------------------------------------------------




GNU General Public License (GPLv3)
See detailed text in license directory 
