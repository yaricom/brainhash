#!/bin/sh
#
# The analyze cyclic
#

for i in `seq 1 $2`;
do
    ./analyze.sh out/intermediate/$1/preprocessed_$i.npy out/analyzer/$1/$1_$i.npy
done 

