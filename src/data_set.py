#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The data set builder

@author: yaric
"""
import os

import pandas as pd
import numpy as np

def loadSessionRecords(folder):
    """
    Load records from provided folder and creates dataframe with flattened numpy arrays loaded
    Arguments:
        folder the folder to look for records
    Return:
        the data frame with loaded flattened arrays as columns
    """
    print("Loading session records from folder: " + folder)
    df = pd.DataFrame()
    for file in os.listdir(folder):
        if file.endswith('.npy'):
            name = file.split(".")[0] # skip file extension
            data = np.load(folder + "/" + file)
            df[name] = data.flatten()
            
    return df

def joinSessionRecords(parent_dir, records):
    """
    Loads and join records with provided records identifiers (the names of folders where 
    numpy arrays saved for specific session/subject)
    Arguments:
        parent_dir the parent forlder
        records the list of subfolders names for specific records
    Return:
        the data frame with all records joined
    """
    df = None
    for rec in records:
        df_rec = loadSessionRecords(parent_dir + "/" + rec)
        if df is None:
            df = df_rec
        else:
            df = df.join(df_rec)
            
    return df

def createDataSet(signal_dir, signal_records, noise_dir, out_dir, join_signals = True):
    """"
    Creates data set and saves it into out_dir folder as list of CSV files
    Arguments:
        signal_dir the parent folder for all signal records
        signal_records the list of signal records identifiers (sessions, subjects, etc)
        noise_dir the folder with noise records data
        out_dir the output directory to hold results
        join_signals the flag to indicate whether signal records should be joined (Default: True)
    """
    if join_signals:
        signal_df = joinSessionRecords(signal_dir, signal_records)
        path = "%s/signal.csv" % (out_dir)
        signal_df.to_csv(path, index = False)
        print("Signal data saved to: " + path)
    else:
        for rec in signal_records:
            signal_df = loadSessionRecords(signal_dir + "/" + rec)
            path = "%s/%s.csv" % (out_dir, rec)
            signal_df.to_csv(path, index = False)
            print("Signal data saved to: " + path)
            
    noise_df = loadSessionRecords(noise_dir)
    path = "%s/noise.csv" % (out_dir)
    noise_df.to_csv(path, index = False)
    print("Noise data saved to: " + path)
            
    
        
    
    

