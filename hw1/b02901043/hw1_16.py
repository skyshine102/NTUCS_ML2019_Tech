#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 21:32:58 2019

@author: rljahn
"""

'''
consider a validation procedure that randomly samples 1000 examples from the training set 
for validation and leaves the other examples for training g svm. 
Fix C = 0.1 and use the validation procedure to choose the best log 10 γ ∈ {−2, −1, 0, 1, 2} according
to E val . If there is a tie of E val , choose the smallest γ. Repeat the procedure 100 times. Plot a
histogram for the number of times each log 10 γ is selected.
'''

from sklearn import svm
import util
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss

# Load datasets
train, test = util.load_data()

# range of parameter C
c = 0.1

# epoch 
epoch = 100

# range of parameter gamma
gamma = [1e-2, 1e-1, 1e0, 1e1, 1e2]

# preprocessing
X,y = util.preprocessing(train, 0.0)


# traing svm and store the distance of any free support vector to the hyperplane in the Z space
best_gamma_count = np.zeros(len(gamma))

for i in range(epoch):
    print(">>>> Now is epoch {}".format(i))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1000)
    E_val = np.zeros(len(gamma))
    for index, g in enumerate(gamma):
        #print(">>>>> gamma = {} >>>".format(g))
        clf = svm.SVC(C = c, kernel = 'rbf', gamma = g)
        clf.fit(X_train, y_train) 
        s = zero_one_loss(y_test, clf.predict(X_test)) # not normalized
        #print(s)
        E_val[index] = s
       
    best_gamma_count[np.argmin(E_val)] += 1
    

# result = array([ 0.,  0.,  6., 74., 20.])
# plot result
plt.bar(np.log10(gamma),best_gamma_count)
plt.xlabel("log10(gamma)")
plt.ylabel("# of votes in each gamma (total 100)")
plt.ylim(0,100)
plt.yticks(np.arange(0,100+10,10))
plt.savefig("hw1_16.png")
plt.show()

# Finding
print("According to the validation procedure, the best gamma is likely to be 10.")
print("However, to decide whether it is the best, we have to compare the resulting error from testing data.") 

