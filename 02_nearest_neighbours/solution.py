# Author: Nína Lea Z. Jónsdóttir
# Date: 25.08.2022
# Project: 02_nerest_neighbours
# Acknowledgements:
# 1.4, 1.5 collaborated with Magnea
# 2.3 - 2.5 collaborated with Arnbjörg

from typing import Concatenate
import numpy as np
import matplotlib.pyplot as plt

from tools import load_iris, split_train_test, plot_points

from sklearn.metrics import accuracy_score


def euclidian_distance(x: np.ndarray, y: np.ndarray) -> float:
    '''
    Calculate the euclidian distance between points x and y
    '''
    d = np.sqrt(np.sum(np.power(x-y,2)))
    return d



def euclidian_distances(x: np.ndarray, points: np.ndarray) -> np.ndarray:
    '''
    Calculate the euclidian distance between x and and many
    points
    '''
    distances = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        d = euclidian_distance(x,points[i])
        # load dinstances into array
        distances[i] = d
    return distances


def k_nearest(x: np.ndarray, points: np.ndarray, k: int):
    '''
    Given a feature vector, find the indexes that correspond
    to the k-nearest feature vectors in points
    '''
    d = euclidian_distances(x,points)
    sort_minimum = np.argsort(d)
    return sort_minimum[:k]


def vote(targets, classes):
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    # combine arrays
    combine_arrays = np.concatenate([targets,classes])
    # get count for each value in array
    value, count = np.unique(combine_arrays, return_counts = True)
    count_index = np.argmax(count)
    return value[count_index]


def knn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    kn = k_nearest(x, points, k)
    voting = vote(point_targets[kn], classes)
    return voting


def remove_one(points: np.ndarray, i: int):
    #Removes the i-th from points and returns
    #the new array
    return np.concatenate((points[0:i], points[i+1:]))

def knn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    # help.remove_one(points, i)

    predictions = np.zeros(points.shape[0]).astype(int)
    for point in range(len(points)):
        x = points[point]
        deconstructed = remove_one(points, point)
        targets = remove_one(point_targets, point)
        kn = int(knn(x,points,point_targets,classes,k))
        predictions[point] = kn

    return predictions.astype(int)


def knn_accuracy(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> float:
    prediction = knn_predict(points, point_targets, classes, k)
    return np.sum(prediction == point_targets)/len(points)


def knn_confusion_matrix(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:

    matrix_size = (len(classes),len(classes))
    conf_matrix = np.zeros(matrix_size)
    # same as in project 1 but instead of guess() we use knn_predict
    for pred,targ in zip(knn_predict(points, point_targets,classes,k),point_targets):
        conf_matrix[pred][targ] += 1

    return conf_matrix


def best_k(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
) -> int:
    best_k=0
    best_acc =0
    for k in range(len(points)-1):
        temp_acc = knn_accuracy(points, point_targets,classes,k+1)
        if temp_acc>best_acc:
                best_acc = temp_acc
                best_k = k+1
    return best_k


def knn_plot_points(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
):
    colours = ['yellow', 'purple', 'blue']
    prediction = knn_predict(points,point_targets,classes, k)
    for i in range(points.shape[0]):
        [x, y] = points[i, :2]

        if (point_targets[i]==prediction[i]):
            plt.scatter(x, y, c=colours[point_targets[i]], edgecolors='green',
                    linewidths=2)
        else:
            plt.scatter(x, y, c=colours[point_targets[i]], edgecolors='red',
                   linewidths=2)
    plt.title('Yellow=0, Purple=1, Blue=2')
    plt.show()



def weighted_vote(
    targets: np.ndarray,
    distances: np.ndarray,
    classes: list
) -> int:
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    # Remove if you don't go for independent section
    ...


def wknn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    # Remove if you don't go for independent section
    ...


def wknn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    # Remove if you don't go for independent section
    ...


def compare_knns(
    points: np.ndarray,
    targets: np.ndarray,
    classes: list
):
    # Remove if you don't go for independent section
    ...


if __name__ == '__main__':
    d, t, classes = load_iris()
    (d_train, t_train), (d_test, t_test) = \
        split_train_test(d, t, 0.8)


    # plot_points(d, t)

    # 1.1
    x, points = d[0,:], d[1:, :]
    dist = euclidian_distance(x, points[50])
    print(dist)

    # 1.2
    ds = euclidian_distances(x, points)
    print(ds)

    # 1.3
    print(k_nearest(x, points, 3))

    # 1.4
    vote_result = vote(np.array([0,0,1,2]), np.array([0,1,2]))
    print(vote_result)

    # 1.5
    x_target, point_targets = t[0], t[1:]
    knn_result = knn(x_target, points, point_targets, classes, 150)
    print(knn_result)

    # 2.1
    d, t, classes = load_iris()
    (d_train, t_train), (d_test, t_test) = split_train_test(d, t, train_ratio = 0.8)
    predictions = knn_predict(d_test, t_test, classes, 5)
    print(predictions)

    # 2.2
    acc = knn_accuracy(d_test, t_test, classes,5)
    print(acc)

    # 2.3
    matrix = knn_confusion_matrix(d_test, t_test, classes, 10)
    print(matrix)

    # 2.4
    k_best = best_k(d_train, t_train, classes)
    print(k_best)

    # 2.5
    knn_plot_points(d, t, classes, 3)
