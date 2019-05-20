import util
import numpy as np
"""
Implement the simple C&RT algorithm without pruning using the Gini impurity as the impurity measure
"""

class TreeNode(object):
    def __init__(self, attribute, threshold):
        self.attr = attribute
        self.thresh = threshold
        self.left = None
        self.right = None
        self.leaf = False
        self.predict = None



def select_threshold(X, y, attribute):
    values = X[attribute]
    # Remove duplicate values by converting the list to a set, then sort the set
    values = list(set(values))
    values.sort()
    max_gini = 1.0
    thresh_val = 0
    # test all threshold values between any two values 
    for i in range(0, len(values) - 1):
        thresh = (values[i] + values[i+1])/2
        gini = gini_impurity(X,y, attribute, thresh)
        if gini > max_gini:
            max_gini = gini
            thresh_val = thresh
    return thresh_val


def gini_impurity(X, y, attribute, threshold):
    '''
    implement weighted gini index
    '''
    classes = list(set(y))
    num_classes = len(set(y))
    sub_1 = y[X[:,attribute] < threshold]
    sub_2 = y[X[:,attribute] > threshold]
    impurity_1 = 1 - sum([ np.mean(sub_1==classes[i])**2 for i in range(num_classes) ])
    impurity_2 = 1 - sum([ np.mean(sub_2==classes[i])**2 for i in range(num_classes) ])
    gini = len(sub_1) * impurity_1 + len(sub_2) * impurity_2 
    return gini



def build_tree(X,y):
	# Get the number of positive and negative examples in the training data
    p = np.sum(y == 1)
    n = np.sum(y == -1)
	# If train data has all positive or all negative values
	# then we have reached the end of our tree
	if p == 0 or n == 0:
		# Create a leaf node indicating it's prediction
		leaf = TreeNode(None,None)
		leaf.leaf = True
		if p > n:
			leaf.predict = 1
		else:
			leaf.predict = 0
		return leaf
	else:
		# Determine attribute and its threshold value with the lowest impurity
		best_attr, threshold = choose_attr(df, cols, predict_attr)
		# Create internal tree node based on attribute and it's threshold
		tree = Node(best_attr, threshold)
		sub_1 = df[df[best_attr] < threshold]
		sub_2 = df[df[best_attr] > threshold]
		# Recursively build left and right subtree
		tree.left = build_tree(sub_1, cols, predict_attr)
		tree.right = build_tree(sub_2, cols, predict_attr)


    return tree


def predict(node, row_df):
	# If we are at a leaf node, return the prediction of the leaf node
	if node.leaf:
		return node.predict
	# Traverse left or right subtree based on instance's data
	if row_df[node.attr] <= node.thres:
		return predict(node.left, row_df)
	elif row_df[node.attr] > node.thres:
		return predict(node.right, row_df)


def test_predictions(root, X, y):
	num_data = len(X)
	num_correct = 0
	for index,row in df.iterrows():
		prediction = predict(root, row)
		if prediction == row['Outcome']:
			num_correct += 1
    return round(num_correct/num_data, 2)



def print_tree(root, level):
	print(counter*" ", end="")
	if root.leaf:
		print(root.predict)
	else:
		print(root.attr)
	if root.left:
		print_tree(root.left, level + 1)
	if root.right:
        print_tree(root.right, level + 1)

def main():

    return


if __name__ == '__main__':

    # load data
    X_train, y_train = util.load_data("hw3_train.dat")
    X_test, y_test = util.load_data("hw3_test.dat")
    print(X_train.shape)
    

    # Q11 draw the resulting tree


    # Q12 E_in, Eout by 0/1 error


    # Q13 simple pruning technique of restricting the maximum tree height. Plot curves of h versus E in (g h ) and h versus E out (g h ) 
    # using the 0/1 error in the same figure. Describe your findings.


    # Bagging algorithm