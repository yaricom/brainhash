#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The raw EEG data preprocessor

@author: yaric
"""

import numpy as np
import pandas as pd
import argparse

import config
import utils

def preprocess(input_file, out_file, dtype = np.float):
    """
    Do input data file preprocessing and save result into intermediate directory
    Arguments:
        input_file the input data file (CSV)
        out_file the output file to store preprocessed results
        dtype the data type for output file (default: np.float)
    """
    df = pd.read_csv(input_file)
    
    print (df.describe())
    
    # the names of columns to be dropped
    drop_columns = ['time', 'primeFreq', 'secondFreq']
    #dropped = df.loc[:, drop_columns]
    
    # drop columns in the data set
    df.drop(drop_columns, axis=1, inplace=True)

    # do min/max scalling in order to get data in range [0, 1]
    dmin, dmax = df.min(), df.max()
    df_norm = (df - dmin) / (dmax - dmin)
    
    # convert to numpy array and save
    X = np.array(df_norm).astype(dtype)
    
    utils.checkParentDir(out_file, clear = False)# Create output directory
    np.save(out_file, X)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The input data preprocessor')
    parser.add_argument('input_file',  
                        help='the input data file as CSV')
    parser.add_argument('--out_file', default=config.preprocessor_out_file, 
                        help='the file to store preprocessed data in custom format')
    args = parser.parse_args()
    
    print("Do preprocessing of: %s and saving results to: %s" % (args.input_file, args.out_file))
    
    preprocess(args.input_file, args.out_file)