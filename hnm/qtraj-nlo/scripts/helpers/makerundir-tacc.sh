#!/bin/bash

if [ "$#" != "1" ]; then
    echo "Usage: makerundir-tacc.sh <run-name>"
    exit
fi

echo "Attempting to create $1"

dir=${WORK}/qtraj-runs
if [ ! -d $dir ]; then
    mkdir $dir
fi

dir=${WORK}/qtraj-runs/${1}
if [ -d $dir ]; then
    echo "Run directory exists, exiting"
    exit
fi

mkdir $dir
cp ./qtraj $dir
cp -r ./input $dir
cp -r ./lib $dir
cp -r ./scripts $dir

ln -s $dir ./$1

echo "Done"
