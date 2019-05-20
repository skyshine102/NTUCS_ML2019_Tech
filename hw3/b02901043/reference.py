#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:18:25 2019

@author: rljahn
"""

import numpy as np
import pandas as pd

class Node:
    def __init__(self, theta, index, value=None):
        self.theta = theta     # 划分的阈值
        self.index = index     # 选用的维度
        self.value = value     # 根节点的值
        self.leftNode = None
        self.rightNode = None
    
def gini(Y):
    l = Y.shape[0]
    if l == 0:
        return 1
    return 1-(np.sum(Y==1)/l)**2-(np.sum(Y==-1)/l)**2


def one_stump(X, Y, thres):
    l = thres.shape[0]
    mini = Y.shape[0]
    for i in range(l):
        Y1 = Y[X<thres[i]]
        Y2 = Y[X>=thres[i]]
        judge = Y1.shape[0]*gini(Y1)+Y2.shape[0]*gini(Y2)
        if mini>judge:
            mini = judge
            b = thres[i]
    return mini, b


def decision_stump(X, Y):
    row, col = X.shape
    Xsort = np.sort(X, 0)
    thres = (np.r_[Xsort[0:1, :]-0.1, Xsort]+np.r_[Xsort, Xsort[-1:, :]+0.1])/2
    mpurity = row; mb = 0; index = 0
    for i in range(col):
        purity, b = one_stump(X[:, i], Y[:, 0], thres[:, i])
        if mpurity > purity:
            mpurity = purity; mb = b; index = i
    return mb, index

def stop_cond(X, Y):
    if np.sum(Y!=Y[0])==0 or X.shape[0]==1 or np.sum(X!=X[0, :])==0:
        return True
    return False


def dTree(X, Y):
    if stop_cond(X, Y):
        node = Node(None, None, Y[0])
        return node
    b, index = decision_stump(X, Y)
    pos1 = X[:, index] < b; pos2 = X[:, index] >= b
    leftX = X[pos1, :]
    leftY = Y[pos1, 0:1]
    rightX = X[pos2, :]
    rightY = Y[pos2, 0:1]
    node = Node(b, index)
    node.leftNode = dTree(leftX, leftY)
    node.rightNode = dTree(rightX, rightY)
    return node


def dTree_one(X, Y):
    b, index = decision_stump(X, Y)
    pos1 = X[:, index] < b; pos2 = X[:, index] >= b
    node = Node(b, index)
    value1 = 1 if np.sign(np.sum(Y[pos1]))>=0 else -1
    value2 = 1 if np.sign(np.sum(Y[pos2]))>=0 else -1
    node.leftNode = Node(None, None, np.array([value1]))
    node.rightNode = Node(None, None, np.array([value2]))
    return node

def predict_one(node, X):
    if node.value is not None:
        return node.value[0]
    thre = node.theta; index = node.index
    if X[index] < thre:
        return predict_one(node.leftNode, X)
    else:
        return predict_one(node.rightNode, X)


def err_fun(X, Y, node):
    row, col = X.shape
    Yhat = np.zeros(Y.shape)
    for i in range(row):
        Yhat[i] = predict_one(node, X[i, :])
    return Yhat, np.sum(Yhat!=Y)/row

def bagging(X, Y):
    row, col = X.shape
    pos = np.random.randint(0, row, (row,))
    return X[pos, :], Y[pos, :]


def loadData(filename):
    data = pd.read_csv(filename, sep='\s+', header=None)
    data = data.as_matrix()
    col, row = data.shape
    X = data[:, 0: row-1]
    Y = data[:, row-1:row]
    return X, Y

def CalInternalNode(treenode):
    num_left = 0
    num_right = 0
    if treenode == None:
        return 0
    if (treenode.leftNode == None) and (treenode.rightNode == None):
        return 0
    if treenode.leftNode != None:
        num_left = CalInternalNode(treenode.leftNode)
    if treenode.rightNode != None:
        num_right = CalInternalNode(treenode.rightNode)
    return num_left + num_right + 1


def print_tree(model, level = 0):
    print(level*"  ", end="")
    if model.leftNode == None:
        print(model.value)
    else:
        print("Att: ", model.index)
        #print("thresh: ", model.theta)
    if model.leftNode:
        print_tree(model.leftNode, level + 1)
    if model.rightNode:
        print_tree(model.rightNode, level + 1)

X, Y = loadData('hw3_train.dat')
Xtest, Ytest = loadData('hw3_test.dat')


node = dTree(X, Y)
# Q14 and Q15
_, ein = err_fun(X, Y, node)
_, eout = err_fun(Xtest, Ytest, node)
print('Ein: ', ein, '\nEout: ', eout)

print_tree(node)

