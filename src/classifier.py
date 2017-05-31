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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# the classifiers names
clf_names = ['RandomForestClassifier',
             'AdaBoostClassifier',
             'DecisionTreeClassifier',
             'GaussianProcessClassifier',
             'MLPClassifier',
             'KNeighborsClassifier',
             'GaussianNB',
             'QuadraticDiscriminantAnalysis',
             'RBF_SVM',
             'LINEAR_SVM']

def printReport(results, n_top = 3):
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

def runClassifier(name, X, y, print_report = False):
    """
    Method to run specific classifier over provided data to build best prediction model
    Arguments:
        name the name of classifier
        X the input data samples
        y the target labels per input samples
    """
    random_state = 42
    if name == 'RandomForestClassifier':
        param_dist = {"max_depth": [3, 5, 8, None],
                      "max_features": [1, 3, 4],
                      "min_samples_split": [2, 3, 4],
                      "min_samples_leaf": [1, 3, 10],
                      "bootstrap": [True, False],
                      "criterion": ["gini", "entropy"],
                      "n_estimators": [10, 20, 50, 100]}
        clf = RandomForestClassifier(n_estimators=20, random_state = random_state)
    elif name == 'AdaBoostClassifier':
        param_dist = {"learning_rate": [0.01, 0.1, 1],
                      "n_estimators": [10, 20, 50, 100]}
        clf = AdaBoostClassifier(random_state = random_state)
    elif name == 'DecisionTreeClassifier':
        param_dist = {"max_depth": [3, 5, 8, None],
                      "min_samples_split": [2, 3, 4],
                      "min_samples_leaf": [1, 3, 10, 20, 30]
                      }
        clf = DecisionTreeClassifier(random_state = random_state)
    elif name == 'GaussianProcessClassifier':
        param_dist = {"warm_start": [True, False]}
        clf = GaussianProcessClassifier(1.0 * RBF(1.0), random_state = random_state)
    elif name == 'MLPClassifier':
        param_dist = {"alpha": [0.0001, 0.001, 0.01],
                      "learning_rate_init": [0.001, 0.01, 0.1, 0.5],
                      "momentum": [0.9, 0.99, 0.999],
                      "solver": ["lbfgs", "sgd", "adam"],
                      "activation" : ["logistic", "tanh", "relu"],
                      "hidden_layer_sizes": [4, 8, 10]}
        clf = MLPClassifier(max_iter = 1000, random_state = random_state)
    elif name == 'KNeighborsClassifier':
        param_dist = {"n_neighbors": [2, 3, 5],
                      "algorithm" : ["ball_tree", "kd_tree", "brute"]}
        clf = KNeighborsClassifier(random_state = random_state)
    elif name == 'GaussianNB':
        param_dist = {"priors": [None]}
        clf = GaussianNB()
    elif name == 'QuadraticDiscriminantAnalysis':
        param_dist = {"priors": [None],
                      "reg_param": [0.0, 0.01, 0.1, 0.9]}
        clf = QuadraticDiscriminantAnalysis()
    elif name == 'RBF_SVM':
        param_dist = {"C": [0.1, 0.5, 1.0],
                      "gamma": [0.1, 0.5, 1.0, 2.0, 3.0, 'auto']}
        clf = SVC(random_state = random_state)
    elif name == 'LINEAR_SVM':
        param_dist = {"C": [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]}
        clf = SVC(random_state = random_state, kernel="linear")
        
        
    cv_results = fitClassifier(X, y, clf, param_dist)
    if print_report:
        printReport(cv_results)
        
    return cv_results
    
def findBestResults(X, y):
    """
    Method to run all classifiers and find best results
    Arguments:
        X the input data samples
        y the target labels per input samples
    """
    results = {}
    for name in clf_names:
        cv_results = runClassifier(name, X, y)
        results[name] = cv_results
        
    # print performance reports per classifier
    print("-----------------------------------")
    for name in clf_names:
        printReport(results[name], n_top = 1)
    
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
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The data samples classifier')
    parser.add_argument('signal_file_csv',  
                        help='the signal data file as CSV')
    parser.add_argument('noise_file_csv',  
                        help='the noise data file as CSV')
    
    args = parser.parse_args()
    
    X, y = loadDataSet(args.signal_file_csv, args.noise_file_csv)
    clzs = np.unique(y)
    print("Data set load complete. Samples [%d], targets [%d], classes [%d]" % 
          (X.shape[0], y.shape[0], clzs.shape[0]))
    findBestResults(X, y)
    