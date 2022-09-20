from typing import Union
import numpy as np

from tools import load_iris, split_train_test


def sigmoid(x: float) -> float:
    '''
    Calculate the sigmoid of x
    '''
    if x < -100:
        return 0.0
    else:
        s = 1/(1+np.exp(-x))
    return s


def d_sigmoid(x: float) -> float:
    '''
    Calculate the derivative of the sigmoid of x.
    '''
    s = sigmoid(x)*(1-sigmoid(x))
    return s


def perceptron(
    x: np.ndarray,
    w: np.ndarray
) -> Union[float, float]:
    '''
    Return the weighted sum of x and w as well as
    the result of applying the sigmoid activation
    to the weighted sum
    '''
    weighted_sum =  np.sum(w@x)
    return weighted_sum, sigmoid(weighted_sum)


def ffnn(
    x: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Computes the output and hidden layer variables for a
    single hidden layer feed-forward neural network.
    '''
    #input pattern
    z0 = np.insert(x, 0, 1, axis=0)
    a1 = np.ndarray((M,))
    z1 = np.ndarray((M+1,))
    z1[0]= 1
    a2= np.ndarray((K,))
    y = np.ndarray((K,))
    W1 = np.transpose(W1)
    W2 = np.transpose(W2)

    for i in range(M):
        a1[i], z1[i+1] = perceptron(z0,W1[i])

    for i in range(K):
        a2[i], y[i] =  perceptron(z1,W2[i])

    return y, z0, z1, a1, a2


def backprop(
    x: np.ndarray,
    target_y: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Perform the backpropagation on given weights W1 and W2
    for the given input pair x, target_y
    '''
    dE1 = np.ndarray((W1.shape))
    dE2 = np.ndarray((W2.shape))
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
    dk = y - target_y
    dj = np.ndarray((M,))
    
    for j in range(M): 
        #sumarize upper layer
        sum = np.sum(W2[j+1,:]*dk)
        for i in range(len(z0)):
            #differentiated output layer * sumarize upper layer
            dj[j] = d_sigmoid(a1[j])*sum
            #errors * lower layer output
            dE1[i,j] = dj[j]* z0[i]
    
    for k in range(K):
        for j in range(len(z1)):
            #error * lower layer output
            dE2[j][k] = dk[k]*z1[j]
    
    return y, dE1, dE2


def train_nn(
    X_train: np.ndarray,
    t_train: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
    iterations: int,
    eta: float
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Train a network by:
    1. forward propagating an input feature through the network
    2. Calculate the error between the prediction the network
    made and the actual target
    3. Backpropagating the error through the network to adjust
    the weights.
    '''
    N = X_train.shape[0]
    E_tot = np.zeros(iterations)
    misclassification_rate = np.zeros(iterations)
    guess = np.zeros_like(t_train)
    
    for i in range(iterations):
        dE1_total = np.zeros_like(W1)
        dE2_total = np.zeros_like(W2)
        error = 0
        for j in range(N):
            # create one-hot target for the feature
            target_y = np.zeros(K)
            target_y[t_train[j]] = 1.0
       
            #get gradient error matrixes and output values
            y, dE1, dE2 = backprop(X_train[j], target_y, M, K, W1, W2)

            #sum gradient error matrices for all x in X_Train
            dE1_total = dE1_total + dE1            
            dE2_total = dE2_total + dE2
            #collect guesses for each x in x_train for given iteration parameters
            guess[j] = np.argmax(y)  
            #calculate cross entropy error for iteration
            error = error + (target_y * np.log(y)) + ((1 - target_y) * np.log(1 - y))
        E_tot[i] = np.sum(-error/N)

        #Calulate misclasssification rate for itereration
        misclassification_rate[i] = np.sum(t_train != guess) / len(t_train)
       
        #update parameters     
        W1 = W1 - eta * dE1_total/N
        W2 = W2 - eta * dE2_total/N
   
    return  W1, W2, E_tot, misclassification_rate, guess


def test_nn(
    X: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> np.ndarray:
    '''
    Return the predictions made by a network for all features
    in the test set X.
    '''
    N = X.shape[0]
    guesses = np.zeros(N)

    for i in range(N): 
            y, z0, z1, a1, a2 = ffnn(X[i], M, K, W1, W2)
            guesses[i]= np.argmax(y)
    return guesses

if __name__ == '__main__':
    features, targets, classes = load_iris()
(train_features, train_targets), (test_features, test_targets) = \
    split_train_test(features, targets)

# 1.1
# s = sigmoid(0.5)
# print(s)

# ds = d_sigmoid(0.2)
# print(ds)

# 1.2
# perc = perceptron(np.array([1.0, 2.3, 1.9]),np.array([0.2,0.3,0.1]))
# print(perc)

# 1.3
# # initialize the random generator to get repeatable results
# np.random.seed(1234)

# # Take one point:
# x = train_features[0, :]
# K = 3 # number of classes
# M = 10
# D = 4
# # Initialize two random weight matrices
# W1 = 2 * np.random.rand(D + 1, M) - 1
# W2 = 2 * np.random.rand(M + 1, K) - 1
# y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
# print(y)
# print(z0)
# print(z1)
# print(a1)
# print(a2)

# 1.4
# # initialize random generator to get predictable results
# np.random.seed(42)

# K = 3  # number of classes
# M = 6
# D = train_features.shape[1]

# x = features[0, :]

# # create one-hot target for the feature
# target_y = np.zeros(K)
# target_y[targets[0]] = 1.0

# # Initialize two random weight matrices
# W1 = 2 * np.random.rand(D + 1, M) - 1
# W2 = 2 * np.random.rand(M + 1, K) - 1

# y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)
# print(y)
# print(dE1)

# 2.1
# initialize the random seed to get predictable results
np.random.seed(1234)

K = 3  # number of classes
M = 6
D = train_features.shape[1]

# Initialize two random weight matrices
W1 = 2 * np.random.rand(D + 1, M) - 1
W2 = 2 * np.random.rand(M + 1, K) - 1
# W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(
#     train_features[:20, :], train_targets[:20], M, K, W1, W2, 500, 0.1)
# print("W1tr = ", W1tr)
# print("W2tr = ", W2tr)
# print("Etotal = ", Etotal)
# print("misclassification_rate = ", misclassification_rate)
# print("last_guesses = ", last_guesses)

# 2.2
guesses = test_nn(X_test, M, K, W1, W2)
print("guesses = ", guesses)