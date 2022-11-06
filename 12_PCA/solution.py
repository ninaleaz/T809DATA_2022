# Author: Nína Lea Z. Jónsdóttir
# Date: 03.11.22
# Project: PCA
# Acknowledgements: Magnea, Arnbjörg
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from tools import load_cancer


def standardize(X: np.ndarray) -> np.ndarray:
    '''
    Standardize an array of shape [N x 1]

    Input arguments:
    * X (np.ndarray): An array of shape [N x 1]

    Returns:
    (np.ndarray): A standardized version of X, also
    of shape [N x 1]
    '''
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_standard = (X-X_mean)/X_std

    return X_standard


def scatter_standardized_dims(
    X: np.ndarray,
    i: int,
    j: int,
):
    '''
    Plots a scatter plot of N points where the n-th point
    has the coordinate (X_ni, X_nj)

    Input arguments:
    * X (np.ndarray): A [N x f] array
    * i (int): The first index
    * j (int): The second index
    '''
    X_sta = standardize(X)
    plt.scatter(X_sta[:,i], X_sta[:,j])


def _scatter_cancer():
    X, y = load_cancer()
    X, y = load_cancer()

    fig = plt.figure()
    fig.suptitle('The cancer dataset scatter for every feature dimension against the first one')
    
    for i in range(X.shape[1]):
        plt.subplot(5, 6, i+1)
        scatter_standardized_dims(X[:,:], 0, i)
        
    plt.show()


def _plot_pca_components():
    X, y = load_cancer()

    X_stand = standardize(X)
    pca = PCA(n_components = 30)
    pca.fit_transform(X_stand)
    fig, axes = plt.subplots(nrows = 3, ncols = 1)
    fig.tight_layout()

    print(pca.components_)

    for i in range(30):
        plt.subplot(5, 6, i + 1)
        plt.plot(pca.components_[i])
        plt.title("PCA" + str(i + 1))

    plt.show()



def _plot_eigen_values():
    X, y = load_cancer()
    X_stand = standardize(X)

    pca = PCA(n_components=30)
    pca.fit_transform(X_stand)

    eig = pca.explained_variance_
    plt.plot(eig)

    plt.xlabel('Eigenvalue index')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.show()


def _plot_log_eigen_values():
    X, y = load_cancer()
    X_stand = standardize(X)
    pca = PCA(n_components = 30)
    pca.fit_transform(X_stand)
    eig = pca.explained_variance_
    logeig = np.log10(eig)
    plt.plot(logeig)

    plt.xlabel('Eigenvalue index')
    plt.ylabel('$\log_{10}$ Eigenvalue')
    plt.grid()
    plt.show()


def _plot_cum_variance():
    X, y = load_cancer()
    X_stand = standardize(X)
    pca = PCA(n_components=30)
    pca.fit_transform(X_stand)
    eig = pca.explained_variance_
    cumsu = np.cumsum(eig)
    plt.title('Cumulatice varience as a function of i')
    plt.plot(cumsu)

    plt.xlabel('Eigenvalue index')
    plt.ylabel('Percentage variance')
    plt.grid()
    plt.show()

# MAIN - muna að stroka út
_plot_pca_components()
#_plot_cum_variance()