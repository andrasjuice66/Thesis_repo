import numpy as np
import torch
from torch.utils.data import TensorDataset
from avalanche.benchmarks.utils import AvalancheDataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


def normalize(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x

def double_shoufle(train, label):
    indices = np.arange(label.shape[0])
    np.random.shuffle(indices)

    train = train[indices]
    label = label[indices]
    return train, label




def get_iris(n_splits=5):
    # df = pd.read_csv("/Users/andrasjoos/PycharmProjects/avalanche_learning/Data/iris_data.csv")
    # df_labels = pd.DataFrame(df, columns=['Iris'])
    # #df_labels = df_labels[1:]
    # df = pd.DataFrame.drop(df, columns=['Iris'])
    # #df = df[1:]
    # print(df)
    # print(df_labels)
    # print(len(df),len(df_labels))
    iris = load_iris()
    df = iris['data']
    df_labels = iris['target']
    df = normalize(df)

    df, df_labels = double_shoufle(df, df_labels)

    kf = KFold(n_splits)
    kf.get_n_splits(df)

    train_datasets = []
    test_datasets = []
    i = 0
    for train_index, test_index in kf.split(df):
        i += 1
        # print("TRAIN:", train_index, "TEST:", test_index)

        X_train, X_test = df[train_index], df[test_index]
        y_train, y_test = df_labels[train_index], df_labels[test_index]

        X_train = torch.tensor(X_train).float()
        X_test = torch.tensor(X_test).float()
        y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        train_av = AvalancheDataset(train_dataset)
        test_av = AvalancheDataset(test_dataset)

        train_datasets.append(train_av)
        test_datasets.append(test_av)

    return train_datasets, test_datasets
