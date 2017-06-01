#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The experiment runner

@author: yaric
"""

import numpy as np

import analyzer as an
import classifier as clf
import data_set as ds
import config as conf
import utils as u

def defaultAnalyzerConfig():
    """
    Creates default analyzer config
    """
    config = {
            'batch_size': 1,
            'learning_rate': 0.1,
            'contraction_level': 0.1,
            'corruption_level': 0.1,
            'n_hidden': 1,
            'training_epochs': 10000,
            'encoder': 'cA',
            'bands': 'all',
            'save_plot': True
            }
    return config

def runEEGAnalyzerWithIDs(ids_list, experiment_name, a_config):
    """"
    Runs analyser over preprocessed records placed in standard location
    Arguments:
        ids_list the list of records IDs to be processed (the name of folder for specific session)
        experiment_name the name of experiment (will be used as parent folder name for results)
        a_config the analyser configuration parameters
    """
    for s_id in ids_list:
        in_path = "%s/%s" % (conf.intermediate_dir, s_id)
        out_path = "%s/%s/%s" % (conf.analyzer_out_dir, experiment_name, s_id)
        f_list = u.listFiles(in_path, '.npy')
        for file in f_list:
            f_id = file.split(".")[0].split("_")[-1]
            out_f = "%s/%s_%s.npy" % (out_path, s_id, f_id)
            in_f = "%s/%s" % (in_path, file)
            runEEGAnalyzer(in_f, out_f, a_config)
    

def runEEGAnalyzer(input_file, out_file, a_config):
    """
    Runs analyzer over preprocessed EEG records 
    Arguments:
        input_file the preprocessed input file
        out_file the output file to store analysis results
        a_config the analyser configuration parameters
    """
    an.analyse(input_file=input_file, out_file=out_file, 
               batch_size=a_config['batch_size'],
               learning_rate=a_config['learning_rate'],
               contraction_level=a_config['contraction_level'],
               corruption_level=a_config['corruption_level'],
               n_hidden=a_config['n_hidden'],
               training_epochs=a_config['training_epochs'],
               encoder=a_config['encoder'],
               bands=a_config['bands'],
               save_plot=a_config['save_plot'])
    
def runClassifier(signal_dir, signal_records, noise_dir, out_suffix):
    """
    Runs classifier over analyzed signal/noise records
    Arguments:
        signal_dir the parent folder for all signal records
        signal_records the list of signal records identifiers (sessions, subjects, etc)
        noise_dir the folder with noise records data
        out_suffix the suffix to append to generated files
    """
    # Generate and save data set
    ds.saveDataSet(signal_dir=signal_dir, 
                   signal_records=signal_records, 
                   noise_dir=noise_dir,
                   out_dir=conf.samples_out_dir,
                   out_suffix=out_suffix,
                   join_signals = True)
    
    # Load data set
    signal_csv_path = "%s/signal_%s.csv" % (conf.samples_out_dir, out_suffix)
    noise_csv_path = "%s/noise_%s.csv" % (conf.samples_out_dir, out_suffix)
    X, y = ds.loadDataSet(signal_csv=signal_csv_path,
                          noise_csv=noise_csv_path)
    
    # Do classification
    clzs = np.unique(y)
    print("Data set load complete. Samples [%d], targets [%d], classes [%d]" % 
          (X.shape[0], y.shape[0], clzs.shape[0]))
    clf.findBestResults(X, y)
    
    

