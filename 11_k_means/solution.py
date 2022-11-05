# Author: Nína Lea Z. Jónsdóttir
# Date: 03.11.22
# Project: K-means clustering and gaussian mixture models
# Acknowledgements: Magnea, Roman, Arnbjörg
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.

import numpy as np
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from typing import Union

from tools import load_iris, image_to_numpy, plot_gmm_results


def distance_matrix(
    X: np.ndarray,
    Mu: np.ndarray
) -> np.ndarray:
    '''
    Returns a matrix of euclidian distances between points in
    X and Mu.

    Input arguments:
    * X (np.ndarray): A [n x f] array of samples
    * Mu (np.ndarray): A [k x f] array of prototypes

    Returns:
    out (np.ndarray): A [n x k] array of euclidian distances
    where out[i, j] is the euclidian distance between X[i, :]
    and Mu[j, :]
    '''
    A = np.empty((X.shape[0], Mu.shape[0]))

    for i in range(X.shape[0]):
        for j in range(Mu.shape[0]):
            A[i, j] = np.linalg.norm(X[i]-Mu[j])
    return A


def determine_r(dist: np.ndarray) -> np.ndarray:
    '''
    Returns a matrix of binary indicators, determining
    assignment of samples to prototypes.

    Input arguments:
    * dist (np.ndarray): A [n x k] array of distances

    Returns:
    out (np.ndarray): A [n x k] array where out[i, j] is
    1 if sample i is closest to prototype j and 0 otherwise.
    '''
    
    return (dist == dist.min(axis=1, keepdims=1)).astype(int)


def determine_j(R: np.ndarray, dist: np.ndarray) -> float:
    '''
    Calculates the value of the objective function given
    arrays of indicators and distances.

    Input arguments:
    * R (np.ndarray): A [n x k] array where out[i, j] is
        1 if sample i is closest to prototype j and 0
        otherwise.
    * dist (np.ndarray): A [n x k] array of distances

    Returns:
    * out (float): The value of the objective function
    '''
    J = 0

    for n in range(R.shape[0]):
        for k in range(R.shape[1]):
            J += R[n,k]*dist[n,k]

    return J/R.shape[0]


def update_Mu(
    Mu: np.ndarray,
    X: np.ndarray,
    R: np.ndarray
) -> np.ndarray:
    '''
    Updates the prototypes, given arrays of current
    prototypes, samples and indicators.

    Input arguments:
    Mu (np.ndarray): A [k x f] array of current prototypes.
    X (np.ndarray): A [n x f] array of samples.
    R (np.ndarray): A [n x k] array of indicators.

    Returns:
    out (np.ndarray): A [k x f] array of updated prototypes.
    '''
    mu = np.zeros((Mu.shape))

    for i in range(Mu.shape[0]):
        num = 0
        den = 0

        for n in range(X.shape[0]):
            num += R[n,i] * X[n]
            den += R[n,i]

        if den==0:
            mu[i] = num
        else:    
            mu[i] = num/den

    return mu


def k_means(
    X: np.ndarray,
    k: int,
    num_its: int
) -> Union[list, np.ndarray, np.ndarray]:
    # We first have to standardize the samples
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_standard = (X-X_mean)/X_std
    # run the k_means algorithm on X_st, not X.

    # we pick K random samples from X as prototypes
    nn = sk.utils.shuffle(range(X_standard.shape[0]))
    Mu = X_standard[nn[0: k], :]

    JS = []

    for n in range(num_its):
        dist = distance_matrix(X_standard, Mu)
        R = determine_r(dist)
        Js = determine_j(R, dist)
        Mu = update_Mu(Mu, X_standard, R)
        JS.append(Js)

    # Then we have to "de-standardize" the prototypes
    for i in range(k):
        Mu[i, :] = Mu[i, :] * X_std + X_mean

    return Mu, R, np.asarray(JS)


def _plot_j():
    X, y, c = load_iris()
    Mu, R, Js = k_means(X, 4, 10)
    plt.figure()
    plt.plot(range(10), Js)
    plt.xlabel("Iterarions")
    plt.ylabel("J")
    plt.show()


def _plot_multi_j():
    X, y, c = load_iris()
    k = [2, 3, 5, 10]

    plt.figure()

    for i in range(len(k)):
        Mu, R, Js= k_means(X, k[i], 10)
        plt.plot(range(10), Js,label = ('k = ' +  str(k[i])))
        
    plt.xlabel("Iterarions")
    plt.ylabel("J")
    plt.legend(loc="upper right")
    plt.show()


def k_means_predict(
    X: np.ndarray,
    t: np.ndarray,
    classes: list,
    num_its: int
) -> np.ndarray:
    '''
    Determine the accuracy and confusion matrix
    of predictions made by k_means on a dataset
    [X, t] where we assume the most common cluster
    for a class label corresponds to that label.

    Input arguments:
    * X (np.ndarray): A [n x f] array of input features
    * t (np.ndarray): A [n] array of target labels
    * classes (list): A list of possible target labels
    * num_its (int): Number of k_means iterations

    Returns:
    * the predictions (list)
    '''
    Mu, R, Js = k_means(X, len(classes), num_its)
    count_0 = np.zeros(len(classes))
    count_1 = np.zeros(len(classes))
    count_2 = np.zeros(len(classes))
    R_pred = np.zeros(len(R))

    for j in range(len(R)):
        R_pred[j] = np.argmax(R[j])

    for i in range(len(R)):

        if R_pred[i] == 0:
            if t[i] == 0: count_0[0] += 1
            elif t[i] == 1: count_0[1] += 1
            elif t[i] == 2: count_0[2] += 1
        elif R_pred[i] == 1:
            if t[i] == 0: count_1[0] += 1
            elif t[i] == 1: count_1[1] += 1
            elif t[i] == 2: count_1[2] += 1
        elif R_pred[i] == 2:
            if t[i] == 0: count_2[0] += 1
            elif t[i] == 1: count_2[1] += 1
            elif t[i] == 2: count_2[2] += 1
            
    for k in range(len(R)):

        if R_pred[k] == 0: R_pred[k] = np.argmax(count_0)
        elif R_pred[k] == 1: R_pred[k] = np.argmax(count_1)
        elif R_pred[k] == 2: R_pred[k] = np.argmax(count_2)

    return R_pred


def _iris_kmeans_accuracy():
    X, y, c = load_iris()

    predictions = k_means_predict(X, y, c, 5)

    print(confusion_matrix(y, predictions))
    print(accuracy_score(y, predictions))


def plot_image_clusters(n_clusters: int):
    '''
    Plot the clusters found using sklearn k-means.
    '''
    image, (w, h) = image_to_numpy()
    
    kmeans = KMeans(n_clusters = n_clusters, random_state = 0).fit(image)

    plt.subplot(121)
    plt.imshow(image.reshape(w, h, 3))
    plt.subplot(122)
    # uncomment the following line to run
    plt.imshow(kmeans.labels_.reshape(w, h), cmap="plasma")
    plt.show()


def _gmm_info():
    model = GaussianMixture(n_components = 3).fit(X)
    
    mix = model.weights_
    mean = model.means_
    cov = model.covariances_

    print('The mixing coefficients', mix)
    print('Mean vectors', mean)
    print('Covariance matrices', cov)


def _plot_gmm():
    model = GaussianMixture(n_components = 3).fit(X)
    pred = model.predict(X)
    mean = model.means_
    cov = model.covariances_
    plot_gmm_results(X, pred, mean, cov)