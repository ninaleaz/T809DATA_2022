# Author: 
# Date:
# Project: 
# Acknowledgements: 
#


from tools import scatter_3d_data, bar_per_axis

import matplotlib.pyplot as plt
import numpy as np


def gen_data(
    n: int,
    k: int,
    mean: np.ndarray,
    var: float
) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''
    ...


def update_sequence_mean(
    mu: np.ndarray,
    x: np.ndarray,
    n: int
) -> np.ndarray:
    '''Performs the mean sequence estimation update
    '''
    ...


def _plot_sequence_estimate():
    data = ...
    estimates = [np.array([0, 0, 0])]
    for i in range(data.shape[0]):
        ...
    plt.plot([e[0] for e in estimates], label='First dimension')
    ...
    plt.legend(loc='upper center')
    plt.show()


def _square_error(y, y_hat):
    ...


def _plot_mean_square_error():
    ...


# Naive solution to the independent question.

def gen_changing_data(
    n: int,
    k: int,
    start_mean: np.ndarray,
    end_mean: np.ndarray,
    var: float
) -> np.ndarray:
    # remove this if you don't go for the independent section
    ...


def _plot_changing_sequence_estimate():
    # remove this if you don't go for the independent section
    ...

#hann prófaði í tíma:
# 1.1
mean = [0,0]
cov = [[1,0], [0,100]]
# ef við gerum þetta 2x þá koma mismunandi tölur því þetta er randnom
np.random.multivariate_normal(mean,cov,10)
# nú fæ ég sömu tölur og áðan með þessari skipun
# seed er til að prófa, viljum það svosem ekki
np.random.seed(1234)
np.random.multivariate_normal(mean,cov,10)

# 1.2 skila á pdf formi wtf

# 1.4
# mu = mean

# from teacher

from scipy.stats import multivariable_normal
rv = multivariable_normal([0.5,-0.2],[[2.0,0.3],[0.3,0.5]])
rv
# gives us the right thing
rv.

rv.pdf([2.2,-1.2])
rv.pdf([1.1,-0.2])
rv.pdf([0.5,-0.2])

np.identity(5)

x = np.identity(5)
x
x.T
