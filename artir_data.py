from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import KFold
from keras.datasets import mnist as normal_mnist
import numpy as np


def countDistinct(arr, n):
    res = 1

    # Pick all elements one by one
    for i in range(1, n):
        j = 0
        for j in range(i):
            if (arr[i] == arr[j]):
                break

        # If not printed earlier, then print it
        if (i == j + 1):
            res += 1

    return res

def flatten_(data_x):
    length = data_x.shape[0]
    X = np.zeros(shape=(data_x.shape[0], 784))
    for i in range(length):
        X[i] = data_x[i].flatten()
        return X


def normalize(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x


def choose_dataset(dataset_name='iris', two_classes=False, classes=1):
    global X, y, test_X, test_y
    if dataset_name == 'iris':
        iris = load_iris()
        X = iris['data']
        y = iris['target']
        if two_classes:
            X = X[:99]
            y = y[:99]
        X = normalize(X)
        test_X = X
        test_y = y

    if dataset_name == 'small_mnist':  # sklearn 8x8 mnist, maybe good for continual learning
        small_mnist = load_digits()
        if two_classes:
            pass
        X = small_mnist.data
        X = normalize(X)
        y = small_mnist.target
        test_X = None
        test_y = None

    if dataset_name == 'mnist':
        (train_X, train_y), (test_X, test_y) = normal_mnist.load_data()
        for i in train_X:
            train_X[i].flatten()
        if two_classes:
            train_X = train_X[:(classes * 1200 - 1)]
            train_y = train_y[:(classes * 1200 - 1)]

        y = train_y
        X = flatten_(train_X)
        test_X_ = flatten_(test_X)
        test_X = test_X_
        X = normalize(X)

    return X, y, test_X, test_y


def create_scenario(isBinary, isKFOLD, X, y, test_X, test_y):
    if isKFOLD:
        pass
    else:
        # sort the data
        idx = np.argsort(y)
        idxt = np.argsort(test_y)
        X = X[idx]
        y = y[idx]
    # if isKFOLD == False:
    #     test_y = test_y[idxt]
    #     test_X = test_X[idxt]

        if isBinary:
            #num_task = countDistinct(y, len(y)) / 2
            num_task = len(set(y))/2
            instance_per_task= int(len(y)/ num_task)
        else:
            #num_task = countDistinct(y, len(y))
            num_task = len(set(y))
            instance_per_task = int(len(y) / num_task)

    return num_task, instance_per_task


def K_Fold(X, n):
    kf = KFold(n_splits=n)
    kf.get_n_splits(X)
    return kf.split(X)
