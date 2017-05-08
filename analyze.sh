#!/bin/sh
#
# The shell script to start analysis
#

/usr/bin/env python3 src/analyzer.py \
        --out_file $2 --batch_size 5 --learning_rate 0.1 \
        --contraction_level 0.1 --corruption_level 0.1 \
        --n_hidden 16 --training_epochs 10000 \
        --encoder cA $1 --save_plot True
