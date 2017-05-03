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

