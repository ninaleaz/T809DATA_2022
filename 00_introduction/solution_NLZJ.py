#      Help that I got:
# 1.2: Got help from examples.py

import numpy as np
import matplotlib.pyplot as plt


def normal(x: np.ndarray, sigma: float, mu: float) -> np.ndarray:
    # Part 1.1
    y = (x-mu)/sigma
    f = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*(np.power(y,2)))
    return f

def plot_normal(sigma: float, mu:float, x_start: float, x_end: float):
    # Part 1.2
    x_rng = np.linspace(x_start, x_end, 500)
    # plot the normal distribution, by using normal function above
    plt.plot(x_rng, normal(x_rng,sigma,mu))
    
#def _plot_three_normals():
    # Part 1.2

def normal_mixture(x: np.ndarray, sigmas: list, mus: list, weights: list):
    # Part 2.1
    # split the long equation into two seperate pieces
    y = weights/(2*np.pi*np.power(sigmas,2))
    # ATH LAGA ekki hægt að mínusa list frá array
    xmus = np.subtract(x,mus)
    z = (np.power(xmus,2))/(2*np.power(sigmas,2))
    # combine the equation
    mixture = y*np.exp(-z)

'''
def _compare_components_and_mixture():
    # Part 2.2

def sample_gaussian_mixture(sigmas: list, mus: list, weights: list, n_samples: int = 500):
    # Part 3.1

def _plot_mixture_and_samples():
    # Part 3.2
'''

if __name__ == '__main__':

    # Part 1.1
    #print(normal(np.array([-1,0,1]), 1, 0))

    # Part 1.2
    # clear the current figure if there is one
    '''
    plt.clf()
    plot_normal(0.5,0,-5,5)
    plot_normal(0.25,1,-5,5)
    plot_normal(1,1.5,-5,5)
    # show the plot
    plt.show()
    '''

    # Part 2.1
    print(normal_mixture(np.linspace(-5, 5, 5), [0.5, 0.25, 1], [0, 1, 1.5], [1/3, 1/3, 1/3]))