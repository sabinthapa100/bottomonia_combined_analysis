#!/bin/bash -l
./scripts/cleandatafiles.sh
./qtraj -stepper 0
mv output output-0
./qtraj -stepper 1
mv output output-1
./qtraj -stepper 2
mv output output-2
