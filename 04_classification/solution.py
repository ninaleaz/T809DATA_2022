# Author: Nína Lea Z Jónsdóttir
# Date: 07.09.22
# Project: Classification
# Acknowledgements: 
# 1.1 collaborated with Magnea


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
    print(mean)
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
    ...


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
    ...


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
        ...
    likelihoods = []
    for i in range(test_features.shape[0]):
        ...
    return np.array(likelihoods)


def predict(likelihoods: np.ndarray):
    '''
    Given an array of shape [num_datapoints x num_classes]
    make a prediction for each datapoint by choosing the
    highest likelihood.

    You should return a [likelihoods.shape[0]] shaped numpy
    array of predictions, e.g. [0, 1, 0, ..., 1, 2]
    '''
    ...


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
    ...

    # MAIN

    # 1.1
    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets, train_ratio=0.6)

    # teacher:
    # 1.1
    # split train test is using random

    # 1.3
    # one point one number, gives output: likeleehood of this point is in what class

    # 1.4
    # for every datapoint
    # given the vector we get, what class is the output likeliest to be?

    # 1.5 
    # how likely is it that this point comes from this distribution

    # 2.1
    # adding probabilty of each class (the extra formula)
    # we are scaling the lielyhood with this number

    # 2.2
    # pdf svara spurningum

    # independent
    # posteriori should give us better resaults
