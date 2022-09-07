# Author: Nína Lea Z. Jónsdóttir
# Date: 02.09.2022
# Project: Sequential Estimation
# Acknowledgements: 
# Independent section collaborated with Arnbjörg
# ATH vantar:
#       1.2 pdf form
#       ræða um idependent section


from tools import scatter_3d_data, bar_per_axis

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def gen_data(
    n: int,
    k: int,
    mean: np.ndarray,
    var: float
) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''
    var = np.identity(k) * np.power(var,2)
    X = np.random.multivariate_normal(mean, var,n)
    return X


def update_sequence_mean(
    mu: np.ndarray,
    x: np.ndarray,
    n: int
) -> np.ndarray:
    '''Performs the mean sequence estimation update
    '''
    update = mu + (x-mu)/n
    mu_update = np.asarray(update)
    return mu_update


def _plot_sequence_estimate():
    data = gen_data(100,3,np.array([0,0,0]),1)
    estimates = [np.array([0, 0, 0])]
    for i in range(data.shape[0]):
        e = update_sequence_mean(estimates[i], data[i], i+1)
        estimates.append(e)
    # plot
    plt.plot([e[0] for e in estimates], label='First dimension')
    plt.plot([e[1] for e in estimates], label='Second dimension')
    plt.plot([e[2] for e in estimates], label='Third dimension')
    plt.legend(loc='upper center')
    plt.title("for X were $x_i \sim N_3([0,0,0], 1)$")
    plt.show()

    return estimates

def _square_error(y, y_hat):
    return np.power(y-y_hat,2)


def _plot_mean_square_error():
    estimates = _plot_sequence_estimate()
    avg_error = []

    for i in range(len(estimates)-1):
        a_error = np.mean(_square_error([0, 0, 0], estimates[i+1]))
        avg_error.append(a_error)

    plt.plot([e for e in avg_error])
    plt.show()


# Naive solution to the independent question.

def gen_changing_data(
    n: int,
    k: int,
    start_mean: np.ndarray,
    end_mean: np.ndarray,
    var: float
) -> np.ndarray:

    d = np.ndarray(shape=(n, k))
    var = np.identity(k) * np.power(var, 2)
    mean = np.linspace(start_mean, end_mean, num = n)
    for i in range(n):
        d[i] = np.random.multivariate_normal(mean[i], var, 1)
    return d, mean


def weighted_update_sequence_mean(
    mu: np.ndarray,
    x: np.ndarray,
    n: int,
    l: int
) -> np.ndarray:
    mu_update = mu + (x-mu)*(1/n + 1/l)
    return np.asarray(mu_update)


def _plot_changing_sequence_estimate():
    estimates = [np.array([0, 0, 0])]
    avg_error = []
    data, true_mean = gen_changing_data(500, 3, [0, 1, -1], [1, -1, 0], 1)

    for i in range(data.shape[0]):
        e = weighted_update_sequence_mean(estimates[i], data[i], i + 1,50)
        estimates.append(e)
        a_error = np.mean(_square_error(true_mean[i], e))
        avg_error.append(a_error)

    plt.figure(0)
    plt.plot([e[0] for e in estimates], label='First dimension')
    plt.plot([e[1] for e in estimates], label='Second dimension')
    plt.plot([e[2] for e in estimates], label='Third dimension')

    plt.legend(loc='upper center')
    plt.show()

    plt.figure(1)
    plt.plot([e for e in avg_error])
    plt.show()

# #hann prófaði í tíma:
# # 1.1
# mean = [0,0]
# cov = [[1,0], [0,100]]
# # ef við gerum þetta 2x þá koma mismunandi tölur því þetta er randnom
# np.random.multivariate_normal(mean,cov,10)
# # nú fæ ég sömu tölur og áðan með þessari skipun
# # seed er til að prófa, viljum það svosem ekki
# np.random.seed(1234)
# np.random.multivariate_normal(mean,cov,10)

# # 1.4
# # mu = mean

# # from teacher

# from scipy.stats import multivariable_normal
# rv = multivariable_normal([0.5,-0.2],[[2.0,0.3],[0.3,0.5]])
# rv
# # gives us the right thing
# rv.

# rv.pdf([2.2,-1.2])
# rv.pdf([1.1,-0.2])
# rv.pdf([0.5,-0.2])

# np.identity(5)

# x = np.identity(5)
# x
# x.T


if __name__ == '__main__':

    # 1.1
    #print(gen_data(2, 3, np.array([0, 1, -1]), 1.3))
    X = gen_data(5, 1, np.array([0.5]), 0.5)
    #print(X)

    # 1.2
    #scatter_3d_data(X)
    #bar_per_axis(X)

    # 1.4
    mean = np.mean(X, 0)
    new_x = gen_data(1, 3, np.array([0, 0, 0]), 1)
    #print(update_sequence_mean(mean, new_x, X.shape[0]))

    # 1.5
    #_plot_sequence_estimate() 

    # 1.6
    #_plot_mean_square_error()

    # Independent Section
    _plot_changing_sequence_estimate()

