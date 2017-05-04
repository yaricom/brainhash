#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The common configuration data

@author: yaric
"""

# The root path to the data directory
data_dir = "../data"
# The output directory
out_dir = "../out"
# The intermediate output directory
intermediate_dir = out_dir + "/intermediate" 
# The analuzer output directory
analyzer_out_dir = out_dir + "/analyzer"

# the preprocessor output file
preprocessor_out_file = intermediate_dir + "/preprocessed_results.npy"

# the analyzer scores output
analyzer_scores_out_file = analyzer_out_dir + "/scores_out.npy"