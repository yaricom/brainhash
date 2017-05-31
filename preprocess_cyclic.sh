#!/bin/sh
#
# The preprocess cyclic
#
#
#
help () {
    echo
    echo "The cyclic preprocessing routine over list of records"
    echo "Usage:"
    echo "      preprocess_cyclic.sh dir prefix end start"
    echo "          dir - the path to the records dir"
    echo "          prefix - the record prefix"
    echo "          end - the last record index"
    echo "          start - the first record index [optional - default: 1]"
    echo
    
}

if [ "$#" -ne 3 ]; then
    help
    exit 0
fi

if [ "$#" -ne 4 ]; then
    START=1
else
    START=$4
fi

for i in `seq $START $3`;
do
    ./preprocess.sh $1/$2_$i.csv $1/preprocessed_$i.npy
done 

