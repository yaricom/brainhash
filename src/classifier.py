#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The results classifier

@author: yaric
"""

import argparse

import numpy as np
import pandas as pd

from time import time

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier

def report(results, n_top=3):
    """
    The Utility function to report best scores
    Arguments:
        results the cv_results_ as acquired from hyper parameter search routine
        n_top the number of top best results to print
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            

def fitClassifier(X, y, clf, param_dist):
    """
    Method to test classifier performance by finfing its best hyperparameters
    Arguments:
        X the input data samples
        y the target labels per input samples
        clf the classifier to be tested
        param_dist the hyper parameters search space
    Return:
        the hyper parameters search results
    """
    grid_search = GridSearchCV(clf, param_grid=param_dist)
    start = time()
    grid_search.fit(X, y)
    
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.cv_results_['params'])))
    return grid_search.cv_results_

def runClassifier(name, X, y):
    """
    Method to run specific classifier over provided data to build best prediction model
    Arguments:
        name the name of classifier
        X the input data samples
        y the target labels per input samples
    """
    if name == 'RandomForestClassifier':
        param_dist = {"max_depth": [3, None],
                      "max_features": [1, 3, 4],
                      "min_samples_split": [2, 3, 4],
                      "min_samples_leaf": [1, 3, 10],
                      "bootstrap": [True, False],
                      "criterion": ["gini", "entropy"],
                      "n_estimators": [20]}
        clf = RandomForestClassifier(n_estimators=20)
    elif name == 'GaussianProcessClassifier':
        param_dist = {"warm_start": [True, False]}
        clf = GaussianProcessClassifier(1.0 * RBF(1.0))
        
    cv_results = fitClassifier(X, y, clf, param_dist)
    report(cv_results)
    
def buildDataSet(signal_csv, noise_csv):
    """
    Method to build data set from provided CSV files
    Arguments:
        signal_csv the CSV file with signal data
        noise_csv the CSV file with noise data
    Returns:
        the tuple (X, y) with data samples and target labels (1 - signal, 0 - noise)
    """
    df_signal = pd.read_csv(signal_csv, index_col=0)
    df_noise = pd.read_csv(noise_csv, index_col=0)
    # combine
    df_data = df_signal.join(df_noise)
    X = np.asarray(df_data).T
    y = np.zeros(df_data.shape[1])
    y[:df_signal.shape[1]] = 1
    return X, y
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The data samples classifier')
    parser.add_argument('signal_file_csv',  
                        help='the signal data file as CSV')
    parser.add_argument('noise_file_csv',  
                        help='the noise data file as CSV')