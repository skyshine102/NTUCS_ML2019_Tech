#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 20:21:44 2019

@author: rljahn
"""

'''
# Q15 Gaussian kernel, gamma = 80,  “0” versus “not 0”
# Plot the the distance of any free support vector to the hyperplane in the (infinite-dimensional) Z
# space versus log 10 C.
# Describe your findings
'''

from sklearn import svm
import util
import numpy as np
import matplotlib.pylab as plt

# Load datasets
train, test = util.load_data()

# range of parameter C
C = [1e-2, 1e-1, 1e0, 1e1, 1e2]
gamma = 80
# preprocessing 
X,y = util.preprocessing(train, 0.0)


# traing svm and store the distance of any free support vector to the hyperplane in the Z space
dist = np.zeros(len(C))
for id, c in enumerate(C):
    print(">>>>> C = {} >>>".format(c))
    clf = svm.SVC(C = c, kernel = 'rbf', gamma = gamma)  # verbose = 1 can print log in Libsvm 
    clf.fit(X, y) 


    # find free support vectors
    alpha = np.abs(clf.dual_coef_)
    index = np.logical_and(alpha < c, alpha > 0)[0]
    #print(index)


    # w = alphas * support vectors
    free_sv = clf.support_vectors_[index] 
    # |w|^2=2*(dual_obj + sum alpha_i).
    # distance = |decision_value| / |w|.
    dual_obj = 0
    
    sum_alpha_i = np.sum( np.abs(clf.dual_coef_))
    #w = np.sqrt(2*(dual_obj + sum_alpha_i))
    Q = util.gaussian_kernel(clf.support_vectors_ ,gamma)
    print("Compute Q completed!")
    w2 = np.dot(clf.dual_coef_, np.dot(Q,clf.dual_coef_.T))
    w = np.sqrt(w2)
    
    decision_value = np.abs(clf.decision_function(free_sv[0].reshape(1,2)))
    
    d = decision_value[0]/w[0][0]
    dist[id] = d



# plot result
plt.plot(np.log10(C), dist)
plt.scatter(np.log10(C), dist)
plt.xlabel("log10(C)")
plt.ylabel("distance (Z space)")
plt.savefig("hw1_15_log.png")
plt.show()


# Findings
print("From the plot, we can see a inverse relation between the log10(C) and distance to the hyperplane.")
print("As stated in the lecture, C is a trade-off of large margin & margin violation.") 
print("As C getting smaller, we want larger margin, and therefore distance to the hyperplane larger.")
print("The trend in the plot is expected.")

'''
# plot result
plt.plot(C, dist)
plt.scatter(C, dist)
plt.xlabel("C")
plt.ylabel("distance (Z space)")
plt.savefig("hw1_15.png")
plt.show()



# plot C * d
tmp = np.array(np.log(C)) * np.array(dist)
plt.plot(tmp, dist)
plt.scatter(tmp, dist)
plt.xlabel("log(C) * dist")
plt.ylabel("distance (Z space)")
plt.savefig("hw1_15_log(C)_dist.png")
plt.show()
 '''
