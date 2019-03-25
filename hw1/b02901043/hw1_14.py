#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 19:44:38 2019

@author: rljahn
"""

'''
# Q14 polynomial kernel, Q=2, “4” versus “not 4”
# plot E_in versus C = { 1e-5, 1e-3, 1e-1, 10, 30}.
# Describe your findings
'''

from sklearn import svm
import util
import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import zero_one_loss

# Load datasets
train, test = util.load_data()

# range of parameter C
C = [ 1e-5, 1e-3, 1e-1, 1e1, 1e3 ]

# preprocessing 
X,y = util.preprocessing(train, 4.0)


# traing svm and store the training loss
E_in = []
for c in C:
    print(">>>>> C = {} >>>".format(c))
    clf = svm.SVC(C = c, kernel = 'poly', gamma = 1, coef0 = 1, degree = 2)
    clf.fit(X, y) 
    s = zero_one_loss(y, clf.predict(X)) # not normalized
    print(s)
    E_in.append(s)

# plot result
plt.plot(np.log10(C),E_in)
plt.scatter(np.log10(C), E_in)
plt.xlabel("log10(C)")
plt.ylabel("E_in")
plt.savefig("hw1_14.png")
plt.show()

# Findings
print("From the plot, we can see that E_in is a constant across different values of C. That is,")
print("although the underlying curves predicted by different models (different C) might be different,")
print("the accuracy is the same on training set. However, to see whether it's overfitting or not, we have to check the results with the testing set.")
print("The E_in > 0 may result from the fact that the data is not linear separable.")

