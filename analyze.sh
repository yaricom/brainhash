#!/bin/sh
#
# The shell script to start analysis
#

/usr/bin/env python3 src/analyzer.py \
        --out_file $2 --batch_size 5 --learning_rate 0.1 \
        --contraction_level 0.1 --corruption_level 0.3 \
        --n_hidden 2 --training_epochs 50000 \
        --encoder cA $1 --save_plot True \
        --bands thetha,alpha_h
