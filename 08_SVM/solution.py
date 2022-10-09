# Author: Nína Lea Z. Jónsdóttir
# Date: 08.10.22
# Project: SVM
# Acknowledgements: Rakel, Magnea, Arnbjörg

from tools import plot_svm_margin, load_cancer
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, precision_score, recall_score

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
import inspect


def _plot_linear_kernel():

    X, t = make_blobs(n_samples = 40, centers = 2)
    clf = svm.SVC(C = 1000, kernel = 'linear')
    clf.fit(X, t)
    plot_svm_margin(clf, X, t)
    plt.show()


def _subplot_svm_margin(
    svc,
    X: np.ndarray,
    t: np.ndarray,
    num_plots: int,
    index: int
):
    # similar to tools.plot_svm_margin but added num_plots and
    # index where num_plots should be the total number of plots
    # and index is the index of the current plot being generated

    plt.scatter(X[:, 0], X[:, 1], c=t, s=30,
                cmap=plt.cm.Paired)

    # plot the decision function
    ax= plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    Z = svc.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(
        XX, YY, Z,
        colors='k', levels=[-1, 0, 1],
        alpha=0.5, linestyles=['--', '-', '--'])

    # plot support vectors
    ax.scatter(
        svc.support_vectors_[:, 0],
        svc.support_vectors_[:, 1],
        s=100, linewidth=1, facecolors='none', edgecolors='k')


def _compare_gamma():
    
    X, t = make_blobs(n_samples = 40, centers = 2, random_state = 6)

    clf=svm.SVC(C=1000, kernel='rbf')
    clf.fit(X, t)
    #print( 1 / (2 * X.var()))
    fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    fig.suptitle('SVM with a radial basis function - Gamma comparison')
    plt.subplot(1,3,1)
    ax1.set_title('gamma = default')
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    _subplot_svm_margin(clf, X, t, 3, 1)

    clf=svm.SVC(C=1000, kernel='rbf', gamma = 0.2)
    clf.fit(X, t)
    #print(clf.support_vectors_)
    plt.subplot(1,3,2)
    ax2.set_title('gamma = 0.2')
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    _subplot_svm_margin(clf, X, t, 3, 2)

    clf=svm.SVC(C=1000, kernel='rbf', gamma = 2)
    clf.fit(X, t)
    #print(clf.decision_function_shape)
    plt.subplot(1,3,3)
    ax3.set_title('gamma = 2')
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    _subplot_svm_margin(clf, X, t, 3, 3)

    plt.subplots_adjust(wspace = 0.6)

    plt.show()


def _compare_C():

    X, t = make_blobs(n_samples=40, centers=2, random_state=6)
 
    clf = svm.SVC(C=1000,  kernel='linear')
    clf.fit(X, t)
    _subplot_svm_margin(clf, X, t, 5, 1)
 
    clf = svm.SVC(C=0.5,  kernel='linear')
    clf.fit(X, t)
    _subplot_svm_margin(clf, X, t, 5, 2)
 
    clf = svm.SVC(C=0.3,  kernel='linear')
    clf.fit(X, t)
    _subplot_svm_margin(clf, X, t, 5, 3)
 
    clf = svm.SVC(C=0.05,  kernel='linear')
    clf.fit(X, t)
    _subplot_svm_margin(clf, X, t, 5, 4)
 
    clf = svm.SVC(C=0.0001,  kernel='linear')
    clf.fit(X, t)
    _subplot_svm_margin(clf, X, t, 5, 5)
 
    plt.show()


def train_test_SVM(
    svc,
    X_train: np.ndarray,
    t_train: np.ndarray,
    X_test: np.ndarray,
    t_test: np.ndarray,
):

    # fit data
    svc.fit(X_train, t_train)
    # predict using test data
    y_pred = svc.predict(X_test)
    # accuracy
    accuracy = sklearn.metrics.accuracy_score(t_test, y_pred)
    # precision
    precision = sklearn.metrics.precision_score(t_test, y_pred)
    # recall
    recall = sklearn.metrics.recall_score(t_test, y_pred)
    return(accuracy,precision,recall)

    

    
