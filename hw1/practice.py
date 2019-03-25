#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 21:41:47 2019

@author: rljahn
"""
import pandas as pd
import numpy as np
from sklearn import svm
from util import * 




# 0/1 error for evaluating E in , E val and E out

print('w = ',clf.coef_)
print('b = ',clf.intercept_)
print('Indices of support vectors = ', clf.support_)
print('Support vectors = ', clf.support_vectors_)
print('Number of support vectors for each class = ', clf.n_support_)
print('Coefficients of the support vector in the decision function = ', np.abs(clf.dual_coef_))



# Exploratory analysis - plot data (Optional)
ax1 = train_data.plot.scatter(x= 1, y= 2, c= 0, cmap='viridis')
ax2 = test_data.plot.scatter(x= 1, y= 2, c= 0, cmap='viridis')




#plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
#plot_svc_decision_function(model);

# check solves with enough numerical precision







from sklearn.metrics import zero_one_loss
X,y = preprocessing(train_data, 4.0)
E_in = []
for c in C:
    clf = svm.SVC(C = c, kernel = 'poly', gamma = 1 ,degree = 2)
    clf.fit(X, y) 
    s = zero_one_loss(y, clf.predict(X), normalize=False)
    print(s)
    E_in.append(s)

plt.plot(np.log10(C),E_in)
plt.scatter(np.log10(C), E_in)
plt.show()

# Try reducing C (try C= 0.001,0.01,0.1). C is the penalty parameter and as C gets bigger, the model tries to reduce the penalty, and so takes more time to train.








# Q15 Gaussian kernel, gamma = 80,  “0” versus “not 0”
C = [1e-2, 1e-1, 1, 10, 20]
# Plot the the distance of any free support vector to the hyperplane in the (infinite-dimensional) Z
# space versus log 10 C.
# Describe your findings


X,y = preprocessing(train_data, 0.0)
dist = []
for c in C:
    clf = svm.SVC(C = c, kernel = 'rbf', gamma=80)
    clf.fit(X, y) 
    index = (clf.dual_coef_ == 1e-2).reshape(-1)
    sv_on boundary = clf.support_vectors_[index,:]
    # clf.sup[clf.dual_coef_ == 1e-2]
    d = clf.dual_coef_
    
    
    print(d)
    dist.append(d)

plt.plot(np.log10(C), dist)
plt.scatter(np.log10(C), dist)
plt.show()



# Q16 validation procedure that randomly samples 1000 examples from the training
# set for validation and leaves the other examples for training g- svm
C = 0.1
# choose the best log 10 γ ∈ {−2, −1, 0, 1, 2} according to E val
# If there is a tie of E val , choose the smallest γ
# Repeat the procedure 100 times. Plot a histogram for the number of times each log 10 γ is selected.
from sklearn import preprocessing, cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

