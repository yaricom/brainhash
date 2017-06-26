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

def saveDataSet(signal_dir, signal_records, noise_dir, out_dir, out_suffix, join_signals = True):
    """"
    Creates data set and saves it into out_dir folder as list of CSV files
    Arguments:
        signal_dir the parent folder for all signal records
        signal_records the list of signal records identifiers (sessions, subjects, etc)
        noise_dir the folder with noise records data or None if only signal data should be processed
        out_dir the output directory to hold results
        out_suffix the suffix to append to generated files
        join_signals the flag to indicate whether signal records should be joined (Default: True)
    """
    if os.path.exists(out_dir) == False:
        os.makedirs(out_dir)
    
    if join_signals:
        signal_df = joinSessionRecords(signal_dir, signal_records)
        path = "%s/signal_%s.csv" % (out_dir, out_suffix)
        signal_df.to_csv(path, index = False)
        print("Signal data saved to: " + path)
    else:
        for rec in signal_records:
            signal_df = loadSessionRecords(signal_dir + "/" + rec)
            path = "%s/%s_%s.csv" % (out_dir, rec, out_suffix)
            signal_df.to_csv(path, index = False)
            print("Signal data saved to: " + path)
        
    if noise_dir is not None:    
        noise_df = loadSessionRecords(noise_dir)
        path = "%s/noise_%s.csv" % (out_dir, out_suffix)
        noise_df.to_csv(path, index = False)
        print("Noise data saved to: " + path)
            

def loadDataSetWithLabels(signal_csv, signal_labels, noise_csv, max_cls_samples = -1):
    """
    Method to load data set from provided CSV files
    Arguments:
        signal_csv the CSV file with signal data
        signal_labels the list with prefixes of column names to be defined as labesl
        noise_csv the CSV file with noise data
        max_cls_samples the maximal number of class samples to include [-1 to include all]
    Returns:
        the tuple (X, y) with data samples and target labels (0 - noise, 1...len(signal_labels) - signals)
    """
    df_signal = pd.read_csv(signal_csv)
    
    print("Loading data set with signal labels %s" % signal_labels)
    df_data = df_signal.T
    X_df, y = None, None
    rows = df_data.index._data.astype(str)
    label = 1
    for lbl in signal_labels:
        selection = rows[np.char.startswith(rows, lbl)]
        if max_cls_samples > 0:
            up_to = min(len(selection), max_cls_samples)
        else:
            up_to = len(selection)
        if X_df is None:
            X_df = pd.DataFrame(df_data.loc[selection[:up_to],:])
            y = np.full((up_to,), label, dtype=float)
        else:
            X_df = X_df.append(df_data.loc[selection[:up_to],:])
            y = np.append(y, np.full((up_to,), label, dtype=float))
        
        print("Added %d rows with label %d starting with row index name %s" % (up_to, label, lbl))
        label += 1
    
    # add noise
    df_noise = pd.read_csv(noise_csv)
    df_noise = df_noise.T
    if max_cls_samples > 0:
        up_to = min(len(df_noise), max_cls_samples)
    else:
        up_to = len(df_noise)
        
    X_df = X_df.append(df_noise.iloc[:up_to,])
    y = np.append(y, np.zeros((up_to,), dtype=float))

    X = np.asarray(X_df)
    return X, y
      
def loadDataSetWithSignals(signal_csv, signal_labels, max_cls_samples = -1):
    """
    Method to load data set from provided CSV files
    Arguments:
        signal_csv the CSV file with signal data
        signal_labels the list with prefixes of column names to be defined as labesl
        max_cls_samples the maximal number of class samples to include [-1 to include all]
    Returns:
        the tuple (X, y) with data samples and target labels (0 - noise, 1...len(signal_labels) - signals)
    """
    df_signal = pd.read_csv(signal_csv)
    
    print("Loading data set with signal labels %s" % signal_labels)
    df_data = df_signal.T
    X_df, y = None, None
    rows = df_data.index._data.astype(str)
    label = 0
    for lbl in signal_labels:
        selection = rows[np.char.startswith(rows, lbl)]
        if max_cls_samples > 0:
            up_to = min(len(selection), max_cls_samples)
        else:
            up_to = len(selection)
        if X_df is None:
            X_df = pd.DataFrame(df_data.loc[selection[:up_to],:])
            y = np.full((up_to,), label, dtype=float)
        else:
            X_df = X_df.append(df_data.loc[selection[:up_to],:])
            y = np.append(y, np.full((up_to,), label, dtype=float))
        
        print("Added %d rows with label %d starting with row index name %s" % (up_to, label, lbl))
        label += 1

    X = np.asarray(X_df)
    return X, y  
    
    
def loadDataSet(signal_csv, noise_csv):
    """
    Method to load data set from provided CSV files
    Arguments:
        signal_csv the CSV file with signal data
        noise_csv the CSV file with noise data
    Returns:
        the tuple (X, y) with data samples and target labels (1 - signal, 0 - noise)
    """
    df_signal = pd.read_csv(signal_csv)
    df_noise = pd.read_csv(noise_csv)
    # combine
    df_data = df_signal.join(df_noise)
    X = np.asarray(df_data).T
    y = np.zeros(df_data.shape[1])
    y[:df_signal.shape[1]] = 1
    return X, y
    
    

