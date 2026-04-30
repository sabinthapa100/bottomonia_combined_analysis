#!/bin/bash

if [ "$#" != "1" ]; then
    echo "Usage: makerundir.sh <run-name>"
    exit
fi

echo "Attempting to create $1"

if [ -d ./$1 ]; then
    echo "Run directory exists, exiting"
    exit
fi

mkdir ./$1
cp ./qtraj ./$1
cp -r ./input ./$1
cp -r ./scripts ./$1

echo "Done"
