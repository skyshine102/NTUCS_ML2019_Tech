import numpy as np
import util
from sklearn.metrics import zero_one_loss
import matplotlib.pylab as plt

def weighted_error_rate(y_true,y_estimated,weights):
    y_true = np.array(y_true)
    y_estimated = np.array(y_estimated)
    return np.dot(weights,y_true != y_estimated)/np.sum(weights)


def update_weights(y_true,y_estimated,weight,epsilon):
    y_true = np.array(y_true)
    y_estimated = np.array(y_estimated)
    idx = (y_true == y_estimated)
    weight[~idx] = weight[~idx] * epsilon
    weight[idx] = weight[idx] / epsilon

def decision_stump(X_train,y_train,weights):
    num_features = X_train.shape[1] 
    Eu_in = 1
    solution = [0,0,0] # s,i, theta

    for i in range(num_features):
        idx = np.argsort(X_train[:,i])
        thresholds = X_train[:,i][idx]
        thresholds = (thresholds[1:] + thresholds[:-1])/2
        thresholds = np.insert(thresholds,0, -np.inf)
        ss = [1, -1]
        for theta in thresholds:
            for s in ss:
                y_estimated = decision_stump_predict(X_train,s,i,theta)
                E = weighted_error_rate(y_train,y_estimated, weights)
                if E < Eu_in:
                    Eu_in = E
                    solution[0] = s
                    solution[1] = i
                    solution[2] = theta
    return solution, Eu_in

def decision_stump_predict(X, s, id_feature, theta):
    r, c = X.shape
    if s == 1:
        return np.sign(X[:,id_feature]-theta)
    else:
        return -np.sign(X[:,id_feature]-theta)

def calculate_ensembled_G(alphas, g_functions, X):
    return np.sign(np.sum( [ alphas[t] * g_functions[t][0] * np.sign(X[:,g_functions[t][1]] - g_functions[t][2]) for t in range(len(alphas)) ],axis=0 ))


def boosted_error(alphas,g_funcs,X_train,y_train):
    return np.sum(y_train != calculate_ensembled_G(alphas,g_funcs,X_train))/len(y_train)

def adaboost(X_train, y_train, base_model, base_model_predict, weights, T):
    alphas = []
    g_functions = []
    min_error = np.inf
    # lists to store required info
    E_in_gt = []
    E_in_Gt = []
    U_t = []
    for step in range(T):
        print(">>> Iteration:{} , sum of weights: {}".format(step+1,np.sum(weights)))
        solution, error = base_model(X_train,y_train, weights)
        print("Parameters = ", solution)
        print("Error: ",error)    
        g_functions.append(solution)
        # calculate epsilon
        epsilon = np.sqrt((1-error)/error)
        print("epsilon = ",epsilon)
        alphas.append(np.log(epsilon))

        min_error = min(error,min_error)
        y_estimated = base_model_predict(X_train,solution[0],solution[1],solution[2])

        E_in_gt.append(zero_one_loss(y_train,y_estimated))

        y_train_estimated_by_ensemble = calculate_ensembled_G(alphas,g_functions,X_train) 
        E_in_Gt.append(zero_one_loss(y_train,y_train_estimated_by_ensemble))
                
        U_t.append(np.sum(weights))
        update_weights(y_train,y_estimated,weights,epsilon)

    #print("Minimum of error is: ", min_error)
    return alphas,g_functions, E_in_gt, E_in_Gt, U_t




def main():
    # Load data and parsing
    train = util.load_data("hw2_adaboost_train.dat.txt")
    X_train,y_train = util.preprocessing(train)
    test = util.load_data("hw2_adaboost_test.dat.txt")
    X_test,y_test = util.preprocessing(test)
    print("The shape of X_train is ({},{})".format(X_train.shape[0],X_train.shape[1]))
    
    # initialize weights = 1/N
    N = len(y_train)
    print("N = ", N)
    
    # initialzie iterations = 300
    T = 300
    
    # Start training Adaboost-Stump
    weights = np.ones(N) * (1/N)
    print("Initial weights =", weights[:5])
    alphas, g_funcs, E_in_gt, E_in_Gt, U_t = adaboost(X_train,y_train,decision_stump,decision_stump_predict,weights,T)
    # plot results
    print(">>>> plot E_in_gt >>>>")
    plt.plot(E_in_gt)
    plt.savefig("Q13.png")
    plt.show()
    print("From the plot, we can see that E_in(g_t) is neither increasing nor decreasing.")
    print("The plot is somewhat like periodic wave.")
    print("It's because in each round of training, reweighting is made for more diverse hypothesis.") 
    print("The diversity results in no guarantee for the performance of g_t.")
    print("E_in_gT = ",E_in_gt[-1])



    print(">>>> plot E_in_Gt >>>>")
    plt.plot(E_in_Gt)
    plt.savefig("Q14.png")
    plt.show()
    print("From the plot, we can see that E_in(G_t) is decreasing.")
    print("It's because with more rounds of training, the ensembled model is using more diversed base models for prediction.")
    print("Therefore, the performace of G_t is getting better.")
    print("From the proof of Q18, we can see that E_in(G_t) will be 0 within O(log(N)) steps, which can be observed in this plot.")
    print("E_in_GT = ",E_in_Gt[-1])
    
    
    print(">>>> plot U_t >>>>")
    plt.plot(U_t)
    plt.savefig("Q15.png")
    plt.show()
    print("From the plot, we can see that U_t is decreasing exponentially.")
    print("Since epsilon_t < 1/2, the result is expected.")
    print("The trend matches the result of Q17.")
    print("U_T = ",U_t[-1])  
    
    
    # 
    E_out_Gt = []
    for step in range(1,T):
        y_test_estimated_by_ensemble = calculate_ensembled_G(alphas[:step],g_funcs[:step], X_test) 
        E_out_Gt.append(zero_one_loss(y_test,y_test_estimated_by_ensemble))
     
    print(">>>> plot E_out_Gt >>>>")
    plt.plot(E_out_Gt)
    plt.savefig("Q16.png")
    print("From the plot, we can see E_out(G_t) is generally decreasing --> then increasing a bit --> then saturating.")
    print("The result shows that we may consider an early stopping scheme by validation due to the saturation.")
    print("E_out_Gt = ",E_out_Gt[-1])

if  __name__ == '__main__':
    main()

