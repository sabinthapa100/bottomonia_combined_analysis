#!/bin/bash -l

command_exists () {
    type "$1" &> /dev/null ;
}

if command_exists module ; then
    module load gnu 
fi

cd test
make clean
make
rm -rf output
rm -rf output*
./qtraj_test
