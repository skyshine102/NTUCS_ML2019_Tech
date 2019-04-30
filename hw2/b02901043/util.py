#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:36:38 2019

@author: rljahn
"""
import pandas as pd
import numpy as np 

def load_data(file,path = ""):
    #path = "/home/rljahn/NTUCS_ML2019_Tech/hw1/"
    # Load data
    data = pd.read_csv(path + file, delim_whitespace=True, header = None)
    return data


def preprocessing(df):
    X = df.copy()
    y = np.array(X.iloc[:,-1])
    X = np.array(X.iloc[:, :-1])
    return X,y
