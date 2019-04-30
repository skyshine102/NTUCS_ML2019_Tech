#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:36:38 2019

@author: rljahn
"""

'''
Adding Bagging(Bootstrap Aggregating) on top of Linear Ridge Regression for Classification
For Q11, Q12
'''

import numpy as np
import util
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import BaggingClassifier # Baggin Classifier
import matplotlib.pylab as plt


# Load data and parsing
data = util.load_data("hw2_lssvm_all.dat.txt")
X,y = util.preprocessing(data)

# add x0 = 1
X = np.insert(X, 0, 1, axis=1)

# test parameter λ = {0.05, 0.5, 5, 50, 500}

lambbda = [0.05, 0.5, 5, 50, 500]

# fit linear ridge regression
E_in = np.zeros(5)
E_out = np.zeros(5)
for it, lb in enumerate(lambbda):
    print(">>>>> λ = {} >>>".format(lb))
    clf = BaggingClassifier(base_estimator=RidgeClassifier(alpha=lb), n_estimators=250, random_state = 1126)# or max_samples = 400 #,oob_score = True)
    #clf = RidgeClassifier(alpha=lb) #alpha: Regularization strength # tol: precision #solver
    clf.fit(X[:400,:], y[:400]) # return coef_ ,intercept_
    s_in = zero_one_loss(y[:400], clf.predict(X[:400,:]))
    s_out = zero_one_loss(y[401:], clf.predict(X[401:,:]))
    print("E_in =",s_in)
    print("E_out =",s_out)
    E_in[it] = s_in
    E_out[it] = s_out


# plot result
plt.plot(np.log10(lambbda),E_in, '.r-')
plt.plot(np.log10(lambbda),E_out,'.b-')
plt.legend(['E_in','E_out'], loc='lower left')
#plt.ylim(0,0.4)
plt.xlabel("log10(λ)")
plt.ylabel("E")
plt.savefig("hw2_11.png")
plt.show()

# Findings
print("Note: Due to the randomness in Bagging, the trends of E_in, E_out with regrards to different lambda will not ")
print("be the same for each run of the program. Here I present the results when I set random_state = 1126 in sklearn.ensemble.BaggingClassifier.")

print(">>> Q11")
print("Among all λ, λ = {},{},{} results in the minimum E_in(g).".format(lambbda[0],lambbda[1],lambbda[2])) 
print("The corresponding E_in(g) is ",np.min(E_in))

print(">>> Q12")
print("Among all λ, λ = {},{},{} results in the minimum E_out(g).".format(lambbda[0],lambbda[1],lambbda[2])) 
print("The corresponding E_out(g) is ",np.min(E_out))

print("From the results, we can see the Bagging gives us a similar result like that in Q9, Q10,")
print("which is expected because our bagging still draw 400 samples.")
print("Proper lambda may lies in the range [0.05,5].")
print("To find the best lambda, we may need to run many times of the program and average the E_in, E_out")
print("for each lambda.")
