#!/bin/sh
# 
# The preprocessor runner
#

/usr/bin/env python3 src/preprocessor.py \
        --out_file out/intermediate/$2 $1
