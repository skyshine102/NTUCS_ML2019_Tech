import numpy as np



def load_data(filename):
    data = np.loadtxt("https://www.csie.ntu.edu.tw/~htlin/course/ml19spring/hw3/"+filename)
    X = data[:,:2]
    y = data[:,2]
    return X, y



def Gini_index(groups, classes):
    """
    best = 0
    worst = 1/k, where k is the number of classes
    """


    index = 0

    #[ for g in groups]
    return  #1 - sum() 

