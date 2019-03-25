#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 19:21:30 2019

@author: rljahn
"""

'''
# Q13 “2” versus “not 2”
# plot |w| versus C = { 1e-5, 1e-3, 1e-1, 10, 30}. 
# Describe your findings
'''

from sklearn import svm
import util
import numpy as np
import matplotlib.pylab as plt


# load datasets
train, test = util.load_data()

# range of parameter C
C = [ 1e-5, 1e-3, 1e-1, 1e1, 1e3 ]

# preprocessing 
X,y = util.preprocessing(train, 2.0)

# traing svm and store the L2-norm of w
w_norm = []

for c in C:
    print(">>>>> C = {} >>>".format(c))
    clf = svm.SVC(C = c, kernel = 'linear')
    clf.fit(X, y)
    print('w = ',clf.coef_)
    print('b = ',clf.intercept_)
    norm = np.linalg.norm(clf.coef_)
    print("|w|", norm)
    w_norm.append(norm) # default L-2 norm

# plot result (|w| v.s. log(C))
plt.plot(np.log10(C),w_norm)
plt.scatter(np.log10(C),w_norm)
plt.xlabel("log10(C)")
plt.ylabel("|w|")
plt.savefig("hw1_13_log.png")
plt.show()
'''
# plot result (|w| v.s. C)
plt.plot(C,w_norm)
plt.scatter(C,w_norm)
plt.xlabel("C")
plt.ylabel("|w|")
plt.savefig("hw1_13.png")
plt.show()
'''
# Findings QQ? 
print("As stated in the lecture, C is a trade-off of large margin & margin violation.") 
print("From the plots, we can see a positive trend between log10(C) and |w|. With 'larger C',\n the optimization problem is less constraint/regularized, therefore the resulting |w| could be larger.")
print("It is also noted that a decrease in |w| at C = 10, but general trend is stated above.")
