#!/bin/sh
#
# The analyze cyclic
#

help () {
    echo
    echo "The auto-encoder cyclic analysis over list of records"
    echo "Usage:"
    echo "      analyze_cyclic.sh prefix end start"
    echo "          prefix - the record prefix"
    echo "          end - the last record index"
    echo "          start - the first record index [optional - default: 1]"
    echo
    
}

if [[ "$#" -lt 2 ]]; then
    help
    exit 0
fi

if [[ "$#" -ne 3 ]]; then
    START=1
else
    START=$3
fi


for i in `seq $START $2`;
do
    ./analyze.sh out/intermediate/$1/preprocessed_$i.npy out/analyzer/$1/$1_$i.npy
done 

