#!/bin/sh
# 
# The preprocessor runner
#

help () {
    echo
    echo "The records preprocessing routine"
    echo "Usage:"
    echo "      preprocess.sh file out"
    echo "          file - the path to the input record file (CSV)"
    echo "          out - the output file path relative to out/intermediate/"
    echo
    
}

if [[ "$#" -lt 2 ]]; then
    help
    exit 0
fi

/usr/bin/env python3 src/preprocessor.py \
        --out_file out/intermediate/$2 $1
