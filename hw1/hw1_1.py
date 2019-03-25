#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 18:16:03 2019

@author: rljahn
"""
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import svm

# making fake data
# from sklearn.datasets import make_blobs 
# we create 40 separable points
# X, y = make_blobs(n_samples=40, centers=2, random_state=6)


# Q1
X = np.array([[1, 0], [0, 1],[0,-1],[-1,0],[0,2],[0,-2],[-2,0]])
y = np.array([-1, -1,-1,1,1,1,1])

def plot(X,y):
    x1 = [i[0] for i in X]
    y1 = [i[1] for i in X]
    plt.scatter(x1,y1, c = y, cmap=plt.cm.Paired)
    #plt.show()


def my_transform(X):
    return np.array([[2*vec[1]**2 - 4*vec[0] + 2, vec[0]**2-2*vec[1]-3] for vec in X])

X_p = my_transform(X)

# C is penalty of parameter C
clf = svm.SVC(kernel='linear', C = 100000.0) #, C=1000.0) # C = 1.0  #,gamma='scale')
clf.fit(X_p, y)  

# get support vectors
print(clf.support_vectors_)
# get indices of support vectors
print(clf.support_) 
# get number of support vectors for each class
print(clf.n_support_ )


def plot_svm_svs(X,y,clf):
    # plot the decision function
    plot(X,y)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k')
    plt.show()
    print("Constants in decision function.")
    print(clf.intercept_)
    if clf.kernel == 'linear':
        print(clf.coef_)
        print("Weights assigned to the features (coefficients in the primal problem)")
    print("The coefficients of dual problem (alpha)")
    print(clf.dual_coef_)
    
plot_svm_svs(X_p,y, clf)


# Q2 use second-order polynomial transformation
# hard-margin SVM --> C = 0
clf = svm.SVC(kernel='poly',degree = 2, gamma =1, coef0 =1 ,C = 1.0)  #, C=1000.0) # C = 1.0  #,gamma='scale')
clf.fit(X, y)  
plot_svm_svs(X,y, clf)
# get support vectors
print(clf.support_vectors_)
# get indices of support vectors
print(clf.support_) 
# get number of support vectors for each class
print(clf.n_support_ )



#from sklearn.metrics import classification_report, confusion_matrix
#print(confusion_matrix(y_test,y_pred))  
#print(classification_report(y_test,y_pred)) 

#svm.SVC(kernel='poly', degree=8)  

#svclassifier = svm.SVC(kernel='rbf')  