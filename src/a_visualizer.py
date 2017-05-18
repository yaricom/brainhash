#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The analyzer output visualizer

@author: yaric
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

import utils


def loadDataFrame(path, name_prefixes, c_size = 16):
    """
    Creates data frame from numpy data arrays within specified directory with given names.
    The loaded arrays with be PCA transformed to reduce dimensions to one and saved into
    data frame in columns with names corresponding to suffixes
    Arguments:
        path the directory path
        name_prefixes the array of names prefixes for arrays to be loaded
        c_size the size of column
    Return:
        the data frame with PCA transformed arrays
    """
    df = pd.DataFrame()
    pca = PCA(n_components = 1)
    for name in name_prefixes:
        file = path + "/" + name + ".npy"
        raw_data = np.load(file)
        data = pca.fit_transform(raw_data)
        df[name] = data.reshape(c_size) # store PCA transformed
        
    return df

def buildDataFrame(path, name_prefixes):
    df = pd.DataFrame()
    for name in name_prefixes:
        file = path + "/" + name + ".npy"
        raw_data = np.load(file)
        c_size = raw_data.shape[0] * raw_data.shape[1]
        df[name] = raw_data.reshape(c_size)
        
    return df
        
    
def visualize(df, mask_correlation = 0.1):
    """
    Builds variety of visualizations for provided data frame
    Arguments:
        df the data frame with data samples
        mask_threshold the minimal correlation value to be included into visualization
    """
    # The scatter plot
    
    
    # The correlations heat map
    plt.figure()
    corr_df = df.corr().abs()
    # Generate a mask for very small values
    mask = np.zeros_like(corr_df, dtype=np.bool)
    corr = np.array(corr_df)
    mask[corr < mask_correlation] = True
    #sns.heatmap(corr_df, annot=True, linewidths=.5, cmap="Oranges")
    #sns.heatmap(corr_df, annot=True, linewidths=.5, cmap="YlOrRd")
    sns.heatmap(corr_df, mask=mask, annot=True, linewidths=.5, cmap="autumn_r")
    plt.title('The correlation map with values > %.2f' % mask_correlation)
 
    

