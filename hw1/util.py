#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 19:21:54 2019

@author: rljahn
"""

import matplotlib.pylab as plt
import numpy as np
import pandas as pd


def load_data():
    # Path of data
    path = "/home/rljahn/NTUCS_ML2019_Tech/hw1/"

    # Load data
    train_data = pd.read_csv(path + "features.train.txt", delim_whitespace=True, header = None)
    test_data = pd.read_csv(path + "features.test.txt", delim_whitespace=True, header = None)
    return train_data, test_data

def preprocessing(df, i):
    X = df.copy()
    X.loc[X[0] != i, 0] = -1.0
    X.loc[X[0] == i, 0] = 1.0
    y = np.array(X[0])
    X = np.array(X.loc[:, 1:2])
    return X,y

def exloratory_analysis(train_data):
    ax1 = train_data.plot.scatter(x= 1, y= 2, c= 0, cmap='viridis')
    plt.show()

    

def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
def gaussian_kernel(x, gamma):
    '''
    x1 is a vector
    '''
    n = x.shape[0]
    K = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            K[i][j] = np.exp(-gamma * np.sum((x[i]-x[j])**2))
    return K 