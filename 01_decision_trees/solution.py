# Author: Nína Lea Z. Jónsdóttir   
# Date: 23.08.2022
# Project: Decision trees
# Acknowledgements: 
# 1.1 = taken from lecture
# 1.2 = taken from lecture
# 2.5 & independent section collaborated with Magnea


from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree

from tools import load_iris, split_train_test

# imported this to be able to use the "accuracy_score" function
from sklearn.metrics import accuracy_score


def prior(targets: np.ndarray, classes: list) -> np.ndarray:
    '''
    Calculate the prior probability of each class type
    given a list of all targets and all class types
    '''
    samples_count = len(targets)
    class_probs = []

    for c in classes:
        class_count = 0
        for t in targets:
            if t == c:
                class_count += 1
        class_probs.append(class_count / samples_count)

    return class_probs


def split_data(
    features: np.ndarray,
    targets: np.ndarray,
    split_feature_index: int,
    theta: float
) -> Union[tuple, tuple]:
    '''
    Split a dataset and targets into two seperate datasets
    where data with split_feature < theta goes to 1 otherwise 2
    '''
    features_1 = features[features[:,split_feature_index] < theta]
    targets_1 = targets[features[:,split_feature_index] < theta]

    features_2 = features[features[:,split_feature_index] >= theta]
    targets_2 = targets[features[:,split_feature_index] >= theta]

    return (features_1, targets_1), (features_2, targets_2)


def gini_impurity(targets: np.ndarray, classes: list) -> float:
    '''
    Calculate:
        i(S_k) = 1/2 * (1 - sum_i P{C_i}**2)
    '''
    i_Sk = 1/2 * (1-np.sum(np.power(prior(targets,classes),2)))
    return i_Sk


def weighted_impurity(
    t1: np.ndarray,
    t2: np.ndarray,
    classes: list
) -> float:
    '''
    Given targets of two branches, return the weighted
    sum of gini branch impurities
    '''
    g1 = gini_impurity(t_1, classes)
    g2 = gini_impurity(t_2, classes)
    n = t1.shape[0] + t2.shape[0]

    weight_impur = (t1.shape[0]*g1)/n + (t2.shape[0]*g2)/n

    return weight_impur


def total_gini_impurity(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    split_feature_index: int,
    theta: float
) -> float:
    '''
    Calculate the gini impurity for a split on split_feature_index
    for a given dataset of features and targets.
    '''
    (f_3, t_3), (f_4, t_4) = split_data(features, targets, 2, 4.65)
    weighted_again = weighted_impurity(t_3, t_4, classes)

    return weighted_again


def brute_best_split(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    num_tries: int
) -> Union[float, int, float]:
    '''
    Find the best split for the given data. Test splitting
    on each feature dimension num_tries times.

    Return the lowest gini impurity, the feature dimension and
    the threshold
    '''
    best_gini, best_dim, best_theta = float("inf"), None, None
    # iterate feature dimensions

    for i in range(features.shape[1]):
        # get the values from each column and find min and max value
        feature_columns = features[:,i]
        min_value = feature_columns.min()
        max_value = feature_columns.max()
        # create the thresholds
        thetas = np.linspace(min_value,max_value,num_tries)
        # iterate thresholds
        for theta in thetas:
            (f_5, t_5), (f_6, t_6) = split_data(features, targets, i, theta)
            gini_impurity = weighted_impurity(t_5, t_6, classes)
            if gini_impurity < best_gini:
                best_gini = gini_impurity
                print(best_gini)
                best_dim = i
                best_theta = theta
    
    return best_gini, best_dim, best_theta


class IrisTreeTrainer:
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        classes: list = [0, 1, 2],
        train_ratio: float = 0.8
    ):
        '''
        train_ratio: The ratio of the Iris dataset that will
        be dedicated to training.
        '''
        (self.train_features, self.train_targets),\
            (self.test_features, self.test_targets) =\
            split_train_test(features, targets, train_ratio)

        self.classes = classes
        self.tree = DecisionTreeClassifier()

    def train(self):
        self.tree.fit(self.train_features,self.train_targets)

    def accuracy(self):
        # use method from decisiontreeclassifier()
        features = self.test_features.argmax(axis=1)
        accuracy = accuracy_score(features,self.test_targets)
        return accuracy

    def plot(self):
        # use method from decisiontreeclassifier()
        plot_tree(self.tree)
        plt.show()

    def plot_progress(self):
        # Independent section
        # Remove this method if you don't go for independent section.
        # initializing the axes

        training_samples = []
        accuracy = []

        for i in range(1,len(self.train_features-1)):
            self.tree.fit(self.train_features[:i],self.train_targets[:i])
            accuracy_guess = self.tree.predict(self.test_features)
            score = accuracy_score(self.test_targets, accuracy_guess)
            accuracy.append(score)
            training_samples.append(i)

        plt.plot(training_samples,accuracy)
        plt.show()
        
        
    def guess(self):
        # use method from decisiontreeclassifier()
        prediction = self.tree.predict(self.test_features)
        return prediction

    def confusion_matrix(self):
        # create 
        matrix_size = (len(classes),len(classes))
        conf_matrix = np.zeros(matrix_size)

        for pred,targ in zip(self.test_targets, self.guess()):
            conf_matrix[pred][targ] += 1

        return conf_matrix



if __name__ == '__main__':

    
    # 1.1
    x = prior([0,2,3,3],[0,1,2,3])
    print(x)
    
    # 1.2
    features, targets, classes = load_iris()
    (f_1, t_1), (f_2, t_2) = split_data(features, targets, 2, 4.65)
    print((len(f_1), len(t_1)), (len(f_2), len(t_2)))

    # 1.3
    impurity = gini_impurity(t_2, classes)
    print(impurity)

    # 1.4
    w_i = weighted_impurity(t_1, t_2, classes)
    print(w_i)

    # 1.5
    tgi = total_gini_impurity(features, targets, classes, 2, 4.65)
    print(tgi)

    # 1.6
    print(brute_best_split(features, targets, classes, 30))
    
    # Part 2
    # 2.1
    dt = IrisTreeTrainer(features, targets, classes=classes)
    dt.train()

    # 2.2
    print(f'The accuracy is: {dt.accuracy()}')

    # 2.3
    dt.plot()

    # 2.4
    print(f'I guessed: {dt.guess()}')

    # 2.5
    print(dt.confusion_matrix())

    #Independent part
    dt = IrisTreeTrainer(features, targets, classes=classes, train_ratio=0.6)
    dt.plot_progress()