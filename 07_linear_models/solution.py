# Author: Nína Lea Z. Jónsdóttir
# Date: 05.10.22
# Project: Linear Models for Regression
# Acknowledgements: Rakel, Magnea, Arnbjörg


import numpy as np
import matplotlib.pyplot as plt

from tools import load_regression_iris
from scipy.stats import multivariate_normal


def mvn_basis(
    features: np.ndarray,
    mu: np.ndarray,
    sigma: float
) -> np.ndarray:
    
    fi = np.zeros((features.shape[0], mu.shape[0]))
    for i in range(features.shape[0]):
        for j in range(mu.shape[0]):
            fi[i,j] = multivariate_normal.pdf(features[i], mu[j,:], sigma)
    return fi


def _plot_mvn():

    plt.plot(fi)
    plt.show()


def max_likelihood_linreg(
    fi: np.ndarray,
    targets: np.ndarray,
    lamda: float
) -> np.ndarray:

    size = fi.shape[1]
    ident = np.identity(size)
    a = fi.T.dot(fi) + lamda * ident
    b = np.linalg.inv(a)
    c = b.dot(fi.T)
    return c.dot(targets)


def linear_model(
    features: np.ndarray,
    mu: np.ndarray,
    sigma: float,
    w: np.ndarray
) -> np.ndarray:

    pred = []
    fi = mvn_basis(features, mu, sigma)
    print(w.shape)
    print(fi[0,:].shape)

    for i in range(fi.shape[0]):
        pred.append(sum(w*fi[i,:]))
    return pred