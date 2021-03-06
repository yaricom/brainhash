#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The common utilities

@author: yaric
"""

import os
import shutil

def checkParentDir(file, clear = True):
    """
    Check if parent directories exists and create it if neccessary.
    Arguments:
        clear if True the contents of parent directory will be cleared as well (default: True).
    """
    p_dir = os.path.dirname(file)
    if os.path.exists(p_dir) == False:
        os.makedirs(p_dir)
    elif clear == True:
        shutil.rmtree(p_dir)
        os.makedirs(p_dir)
        
def buildDataSetFileNames(name_prefix, suffix_range):
    """
    Creates list of data set file names given prefix and range of indices. The
    file name will be in form: name_prefix_0, name_prefix_1, etc
    Arguments:
        name_prefix the prefix to start file name with
        suffix_range the range of index suffixes to append to file name
    Return:
        the list of file names creted from provided suffix and indexes range
    """
    names = []
    for i in suffix_range:
        names.append(name_prefix + str(i))
        
    return names

def listFiles(path, extension):
    """
    List all files under specified path with given extension
    Arguments:
        path the path to files folder
        extension the file extension for files to be included
    """
    f_list = []
    for file in os.listdir(path):
        if file.endswith(extension):
            f_list.append(file)
            
    return f_list

def stripOutliers(df, multiplier = 2.0):
    """
    Strips outliers from provided data set
    Arguments:
        df the data frame to process
        multiplier the std multiplier to be used for outliers detection
    """
    mean = df.mean()
    std = df.std()
    df[df < (mean - multiplier * std)] = 0
    df[df > (mean + multiplier * std)] = 0
    return df

