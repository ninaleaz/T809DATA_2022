# Author: Nína Lea Z Jónsdóttir
# Date: 07.09.22
# Project: Classification
# Acknowledgements: 
# 1.1, 1.2 collaborated with Magnea


from decimal import ConversionSyntax
from tools import load_iris, split_train_test

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def mean_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the mean of a selected class given all features
    and targets in a dataset
    '''
    class_features = []
    for i in range(len(targets)):
        if targets[i] == selected_class:
            class_features.append(features[i])

    mean = np.mean(class_features, axis = 0)

    return mean


def covar_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the covariance of a selected class given all
    features and targets in a dataset
    '''
    class_features = []
    for i in range(len(targets)):
        if targets[i] == selected_class:
            class_features.append(features[i])

    covar = np.cov(class_features, rowvar = False)

    return covar


def likelihood_of_class(
    feature: np.ndarray,
    class_mean: np.ndarray,
    class_covar: np.ndarray
) -> float:
    '''
    Estimate the likelihood that a sample is drawn
    from a multivariate normal distribution, given the mean
    and covariance of the distribution.
    '''
    likelihood = multivariate_normal(mean = class_mean, cov=class_covar,allow_singular=True).pdf(feature)
    return likelihood


def maximum_likelihood(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum likelihood for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs = [], []
    for class_label in classes:
        means.append(mean_of_class(train_features, train_targets, class_label))
        covs.append(covar_of_class(train_features, train_targets, class_label))
    
    likelihoods = []
    for i in range(test_features.shape[0]):
        class_likelihoods = []
        for j in range(len(classes)):
            class_likelihoods.append(likelihood_of_class(test_features[i],means[j],covs[j]))
        likelihoods.append(class_likelihoods)

    return np.array(likelihoods)


def predict(likelihoods: np.ndarray):
    '''
    Given an array of shape [num_datapoints x num_classes]
    make a prediction for each datapoint by choosing the
    highest likelihood.

    You should return a [likelihoods.shape[0]] shaped numpy
    array of predictions, e.g. [0, 1, 0, ..., 1, 2]
    '''
    class_pred = []
    for i in range(likelihoods.shape[0]):
        pred = np.argmax(likelihoods[i])
        class_pred.append(pred)

    return np.array(class_pred)


def maximum_aposteriori(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum a posteriori for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs, likelihoods, a_post = [], [], [], []
    for class_label in classes:
        means.append(mean_of_class(train_features, train_targets, class_label))
        covs.append(covar_of_class(train_features, train_targets, class_label))
        likelihoods.append((np.count_nonzero(train_targets == class_label)/train_targets.shape[0]))
    #print(likelihoods)
    for i in range(test_features.shape[0]):
        class_likelihoods = []
        for j in range(len(classes)):
            class_likelihoods.append(likelihoods[j]*likelihood_of_class(test_features[i],means[j],covs[j]))
        a_post.append(class_likelihoods)

    return np.array(a_post)

def accuracy(
    likelihoods: np.ndarray,
    test_targets: np.ndarray,
) -> float:
    pred = predict(likelihoods)
    return np.sum(pred == test_targets)/len(test_targets)

def confusion_matrix(
    likelihoods: np.ndarray,
    test_targets: np.ndarray,
    classes: list,
) -> np.ndarray:

    prediction = predict(likelihoods)
    matrix = np.zeros((len(classes), len(classes)))
    for pred, targ in zip(prediction, test_targets):
        matrix[pred][targ] += 1
    return matrix


if __name__ == '__main__':

    # 1.1
    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets, train_ratio=0.6)
    #print(mean_of_class(train_features, train_targets, 0))

    # 1.2
    #print(covar_of_class(train_features, train_targets, 0))

    # 1.3
    class_mean = mean_of_class(train_features, train_targets, 0)
    class_cov = covar_of_class(train_features, train_targets, 0)
    #print(likelihood_of_class(test_features[0, :], class_mean, class_cov))

    # 1.4
    #print(maximum_likelihood(train_features, train_targets, test_features, classes))

    # 1.5 
    likelihoods1 = maximum_likelihood(train_features, train_targets, test_features, classes)
    #print(predict(likelihoods1))

    # 2.1
    likelihoods2 = maximum_aposteriori(train_features, train_targets, test_features, classes)
    #print(likelihoods2)

    # 2.2
    print(accuracy(likelihoods1,test_targets))
    print(accuracy(likelihoods2,test_targets))
    print(confusion_matrix(likelihoods1,test_targets, classes))
    print(confusion_matrix(likelihoods2,test_targets, classes))
