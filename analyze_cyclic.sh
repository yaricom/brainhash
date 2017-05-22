#!/bin/sh
#
# The analyze cyclic
#

if [ "$#" -ne 3 ]; then
    START=1
else
    START=$3
fi


for i in `seq $START $2`;
do
    ./analyze.sh out/intermediate/$1/preprocessed_$i.npy out/analyzer/$1/$1_$i.npy
done 

