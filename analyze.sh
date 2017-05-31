#!/bin/sh
#
# The shell script to start analysis
#

help () {
    echo
    echo "The auto-encoder cyclic analysis of EEG record"
    echo "Usage:"
    echo "      preprocess.sh file out"
    echo "          file - the preprocessed input record file (.npy)"
    echo "          out - the output file path"
    echo
    
}

if [[ "$#" -lt 2 ]]; then
    help
    exit 0
fi

/usr/bin/env python3 src/analyzer.py \
        --out_file $2 --batch_size 5 --learning_rate 0.1 \
        --contraction_level 0.1 --corruption_level 0.3 \
        --n_hidden 2 --training_epochs 50000 \
        --encoder cA $1 --save_plot True \
        --bands delta,alpha_h
