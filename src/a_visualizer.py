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
import matplotlib as plt


def loadDataFrame(path, name_suffixes):
    """
    Creates data frame from numpy data arrays within specified directory with given names.
    The loaded arrays with be PCA transformed to reduce dimensions to one and saved into
    data frame in columns with names corresponding to suffixes
    Arguments:
        path the directory path
        name_suffixes the array of names suffixes for arrays to be loaded
    Return:
        the data frame with PCA transformed arrays
    """
    df = pd.DataFrame()
    pca = PCA(n_components = 1)
    for name in name_suffixes:
        file = path + "/" + name
        raw_data = np.load(file)
        df[name] = pca.fit_transform(raw_data) # store PCA transformed
        
    return df
    
    
def visualize(df):
    """
    Builds variety of visualizations for provided data frame
    Arguments:
        df the data frame with data samples
    """
    # The scatter plot
    
    
    # The correlations heat map
    plt.figure()
    corr_df = df.corr().abs()
    #sns.heatmap(corr_df, annot=True, linewidths=.5, cmap="Oranges")
    #sns.heatmap(corr_df, annot=True, linewidths=.5, cmap="YlOrRd")
    sns.heatmap(corr_df, annot=True, linewidths=.5, cmap="autumn_r")
    plt.title('The correlation map')
