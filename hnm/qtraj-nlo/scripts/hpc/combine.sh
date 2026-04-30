#!/usr/bin/bash

####################################################
# Combines datafiles from scanImpactParameters.pbs.
# Should be executed in the directory containing
# the output folders "output-*".  User must
# manually sync the blist variable.  Script creates
# a tgz file called combined.tgz as output.
####################################################

# impact parameter list
blist=("0" "2.32326" "4.24791" "6.00746" "7.77937" "9.21228" "10.4493" "11.5541" "12.5619" "13.4945" "14.3815")
nb=$((${#blist[@]}))

rm combined.tgz
rm -r combined
mkdir combined

for  (( j=0; j<$nb; j++ ))
do
    b=${blist[$j]}
    echo "Processing ${b}"
    [ -d "combined/${b}" ] && echo "Directory combined/${b}$ exists." || mkdir combined/${b}
    cat output*/${b}/ratios.gz > combined/${b}/ratios.gz
    zcat combined/${b}/ratios.gz | wc
done

tar -zcf combined.tgz combined
rm -r combined
