#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 16:06:27 2019

@author: rljahn
"""

"""
Implement the simple C&RT algorithm without pruning using the Gini impurity as the impurity measure
"""

import numpy as np
import util
import matplotlib.pylab as plt


class TreeNode(object):
    def __init__(self, attribute, threshold):
        self.attr = attribute
        self.thresh = threshold
        self.left = None
        self.right = None
        self.isleaf = False
        self.predict = None
        self.height = None

def gini_impurity(sub_y):
    if len(sub_y) == 0:
        return 1
    else:
        sub_y = np.array(sub_y)
        classes = list(set(sub_y))
        num_classes = len(set(sub_y))
        return 1 - sum([ np.mean(sub_y==classes[i])**2 for i in range(num_classes) ])

def CalLoss2y(y_true, y_predicted):
    loss = np.mean( y_predicted != y_true)
    return loss

def CalLoss(y_true, y_matrix, t):
    tmp = np.sum(y_matrix[:,:t-1], axis = 1)
    tmp[tmp >= 0] = 1
    tmp[tmp < 0] = -1
    y_predicted = tmp
    loss = np.mean([ y_predicted[i] != y_true[i] for i in range(len(y_true)) ])
    return loss

'''
def CalInternalNode(treenode):
    num_left = 0
    num_right = 0
    if treenode == None:
        return 0
    if (treenode.left == None) and (treenode.right == None):
        return 0
    if treenode.left != None:
        num_left = CalInternalNode(treenode.left)
    if treenode.right != None:
        num_right = CalInternalNode(treenode.right)
    return num_left + num_right + 1
'''

def TrainDecisionStump(sorted_Xi, sorted_y):
    #sorted_X = np.array(list(set(sorted_Xi)))
    # why this would make different results??? 
    sorted_X = sorted_Xi
    midpoints = (sorted_X[:-1] + sorted_X[1:])/2

    gini = np.inf 
    best_midpoint = None
    for mp in midpoints:
        left = sorted_y[sorted_Xi < mp]
        right = sorted_y[sorted_Xi >= mp]
        tmp_gini = left.shape[0] * gini_impurity(left) + right.shape[0] * gini_impurity(right)
        if tmp_gini < gini:
            gini = tmp_gini
            best_midpoint = mp
    return gini, best_midpoint


def SplitByDecisionStump(X,y):
    num_attr = X.shape[1]
    # sort X along each attributes
    sort_id = [np.argsort(X[:,i]) for i in range(num_attr)]
    
    Gini = np.inf
    best_attr = None
    best_midpoint = None
    for i in range(num_attr):
        sorted_Xi = X[sort_id[i],i]
        sorted_y = y[sort_id[i]]
        gini, bestmp = TrainDecisionStump(sorted_Xi,sorted_y)
        if gini < Gini:
            Gini = gini
            best_attr = i
            best_midpoint = bestmp
    left_X = X[X[:,best_attr] < best_midpoint]
    left_y = y[X[:,best_attr] < best_midpoint]
    right_X = X[X[:,best_attr] >= best_midpoint]
    right_y = y[X[:,best_attr] >= best_midpoint]
    return left_X, left_y, right_X, right_y , best_attr, best_midpoint
    
def CART(X, y, h = 0, max_H = None):
    # if y is already pure, either all y = 1 or y = -1
    # Note that when all xn are the same should terminate as well.
    if gini_impurity(y) == 0 or X.shape[0] ==1 or np.sum(X != X[0,:]) == 0:
        leaf = TreeNode(None,None)
        leaf.isleaf = True
        leaf.predict = y[0]
        leaf.height = h
        return leaf

    elif h == max_H:
        leaf = TreeNode(None,None)
        leaf.isleaf = True
        boolean = sum(y==1) >= sum(y ==-1)
        if boolean == True:
            leaf.predict = 1
        else:
            leaf.predict = -1
        leaf.height = h
        return leaf
    
    else:
        left_X, left_y, right_X, right_y, best_attr, best_midpoint = SplitByDecisionStump(X,y)
        Node = TreeNode(best_attr, best_midpoint)
        Node.left = CART(left_X,left_y, h+1, max_H = max_H)
        Node.right = CART(right_X,right_y, h+1, max_H= max_H)
        Node.height = h
        return Node


def CART_predict(tree, x):
    '''
    predict the class of a single x
    '''
    if tree.attr == None:
        return tree.predict
    elif x[tree.attr] < tree.thresh:
        return CART_predict(tree.left,x)
    else:
        return CART_predict(tree.right,x)


def BootStrap(X,y, percentage = 0.8):
    index = range(len(X))
    index_chose = np.random.choice(index, size = int(np.floor(percentage * len(X))) )
    sampled_X = X[index_chose]
    sampled_y = y[index_chose]
    return sampled_X, sampled_y


def RandomForest(X, y, T = 30):
    #first_t_tree_loss = np.zeros(T)
    forest = [-1 for i in range(T)]
    for t in range(T):
        if (t % 100 ==0):
            print(t)
        sampled_X, sampled_y = BootStrap(X,y)
        model = CART(sampled_X,sampled_y)
        forest[t] = model
        #first_t_tree_loss[t] = CalLoss(RandomForest_predictv2, forest, X, y)
    return forest #first_t_tree_loss, forest #loss/T, 

def RandomForest_predictv3(forest, X):
    y_matrix = np.zeros((X.shape[0],len(forest)))
    for index, tree in enumerate(forest):
        y_matrix[:,index] = np.array([CART_predict(tree, X[i,:]) for i in range(X.shape[0])])
    
    tmp = np.sum(y_matrix, axis = 1)
    tmp[tmp >= 0] = 1
    tmp[tmp < 0] = -1
    predicted_y_array = tmp
    return y_matrix, predicted_y_array



def print_tree(model, level = 0):
    
    print(level*"  ", end="")
    
    if model.isleaf:
        print(model.predict)
    else:
        print("Att: ", model.attr, sep = '')
        #print("thresh: ", model.thresh)
    if model.left:
        print_tree(model.left, level + 1)
    if model.right:
        print_tree(model.right, level + 1)


def get_max_height(model):
    max_height = -np.inf
    if model.isleaf:
        if model.height > max_height:
            max_height = model.height
    else:
        tmpl = get_max_height(model.left)
        tmpr = get_max_height(model.right)
        if tmpl > tmpr:
            if tmpl > max_height:
                max_height = tmpl
        else:
            if tmpr > max_height:
                max_height = tmpr
    return max_height

if __name__ == '__main__':

    # load data
    print('>>>> Load data >>>>')
    X_train, y_train = util.load_data("hw3_train.dat")
    X_test, y_test = util.load_data("hw3_test.dat")

    # build CART decision tree
    print('>>>> Build CART decision tree >>>>')
    root = CART(X_train,y_train)
    print('>>>> Q11: Draw decision tree >>>>')
    print_tree(root)
    print('\n\n')
    
    print('>>>> Q12: Calculate Ein, Eout >>>>')
    yin = np.array([CART_predict(root, X_train[i,:]) for i in range(X_train.shape[0])])
    Ein = CalLoss2y(y_train, yin)
    yout = np.array([CART_predict(root, X_test[i,:]) for i in range(X_test.shape[0])])
    Eout = CalLoss2y(y_test,yout)
    print('Ein = {}, Eout = {}'.format(Ein,Eout))

    
    print('>>>> Q13: Plot h versus Ein, Eout with pruning >>>>')
    max_height = get_max_height(root)
    print("Max Height is ", max_height)
    Ein_array = np.zeros(max_height+1)
    Eout_array =  np.zeros(max_height+1)
    
    for h in range(0,max_height+1):
        root_tmp = CART(X_train,y_train, max_H = h)
        yin = np.array([CART_predict(root_tmp, X_train[i,:]) for i in range(X_train.shape[0])])
        Ein_array[h] = CalLoss2y(y_train, yin)
        yout = np.array([CART_predict(root_tmp, X_test[i,:]) for i in range(X_test.shape[0])])
        Eout_array[h] = CalLoss2y(y_test,yout)
    
    plt.plot(range(1,max_height+2),Ein_array, label = 'Ein' )
    plt.plot(range(1,max_height+2),Eout_array, label = 'Eout' )
    plt.legend()
    plt.savefig("Q13plot.png")
    plt.show(block=False)
    print("General trend: both Ein and Eout decreases with h. However, when h = 6, which is the maximal height, Eout increases", end = '')
    print(", indicating overfitting. Therefore pruning technique is important in tree methods.")
    
    print('>>>>> Q14 Plot a histogram of Ein(gt) >>>>')
    T = 30000
    forest = RandomForest(X_train, y_train, T= T)
    
    y_matrix, _ = RandomForest_predictv3(forest, X_train)
    
    Ein_gt = [CalLoss2y(y_train,y_matrix[:,i]) for i in range(y_matrix.shape[1])  ]
    
    plt.hist(Ein_gt)
    plt.savefig("Q14plot.png")
    plt.show(block=False)
    
    print('>>>>> Q15 Plot a curve of t versus Ein (Gt) >>>>')
    
    first_t_loss = [ CalLoss(y_train,y_matrix,i) for i in range(1,y_matrix.shape[1]+1)]
    #print("Ein(Gt) reaches 0 as t = {}".format(np.argwhere(first_t_loss==0)[0][0]))
    plt.plot(range(1,T+1), first_t_loss)
    plt.savefig("Q15plot.png")
    plt.show(block=False)



    print('>>>>> Q16 Plot a curve of t versus Eout (Gt) >>>>')
    print("Generate new predictions on testing sets... Plz wait!")
    y_matrix_out, _ = RandomForest_predictv3(forest, X_test)  
    
    first_t_tree_Eout = [ CalLoss(y_test,y_matrix_out,i) for i in range(1,y_matrix_out.shape[1]+1)]
    
    #print("Eout(Gt) reaches 0.07 as t = {}".format(np.argwhere(first_t_tree_Eout==0.07)[0][0]))
    print("The last Eout: ",first_t_tree_Eout[-1])
    plt.plot(range(1,T+1), first_t_tree_Eout)
    plt.savefig("Q16plot.png")
    plt.show(block=False)
    
    # plot 15,16 together
    plt.plot(range(1,1000+1), first_t_loss[:1000], label = "Ein")
    plt.plot(range(1,1000+1), first_t_tree_Eout[:1000], label = "Eout")
    plt.legend()
    plt.savefig("Q1516plot.png")
    plt.show(block=False)
    print("To better visualize, the curves of t versus Ein(Gt), Eout(Gy), t<= 1000 are plotted.")
    print("From the plot we can find that with curve of Eout is above that of Ein, with almost constant difference")
    print("when t > 200. Both curves stay constant when t is large. There is no sign of overfitting. The result shows the power of Bagging - reduce variance.")
    
    
    