import torch
from torch import nn
from torch.optim import SGD, Adam
# from torch.utils.data.dataset import Subset
import torch.nn.utils.rnn as rnn
#import torch.nn.functional as F
from model import face_classifier
# from params import *
import numpy as np


import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from torch.utils.data import DataLoader, TensorDataset

import seaborn as sns
from torchsummary import summary

PRINT_EPOCH = 2


def print_cmx(y_true, y_pred):
    y_true_np = y_true.to('cpu').detach().numpy().copy()
    y_pred_np = y_pred.to('cpu').detach().numpy().copy()
    labels = sorted(list(set(y_true_np)))
    cmx_data = confusion_matrix(y_true_np, y_pred_np, labels=labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cmx, annot=True)
    plt.show()


def train():
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_x, train_t = torch.load('./classifier_train_data.pkl')

    # BATCH_SIZE=1

    # CVのための分割点決め
    n_samples = len(train_x)
    train_size = n_samples * 9 // 10
    test_size = n_samples - train_size

    #ds = TensorDataset(train_x, train_t)
    # ds_train, ds_test = torch.utils.data.random_split(
    #        ds, [train_size, test_size])

    train_x_ndarray = train_x.to('cpu').detach().numpy().copy()
    train_x_ndarray_ = []
    for hatuwa in train_x_ndarray:
        train_x_ndarray_.append(np.mean(hatuwa, axis=0))
    train_x_ndarray = train_x_ndarray_

    train_t_ndarray = train_t.to('cpu').detach().numpy().copy()

    TEST_SIZE = 0.1
    mindist = 5
    minseed = 6900
    minzero = 15
    minone = 20
    mintwo = 20
#
#    for x in range(50000, 100000):
#        X_train, X_test, t_train, t_test = train_test_split(
#            train_x_ndarray, train_t_ndarray, random_state=x, test_size=TEST_SIZE)
#        zero = np.count_nonzero(t_test == 0)
#        one = np.count_nonzero(t_test == 1)
#        two = np.count_nonzero(t_test == 2)
#        M = max([zero, one, two])
#        m = min([zero, one, two])
#        if M - m < mindist:
#            minseed = x
#            minzero = zero
#            minone = one
#            mintwo = two
#            mindist = M-m
#        if x % 100 == 0:
#            print(x)
    X_train, X_test, t_train, t_test = train_test_split(
        train_x_ndarray, train_t_ndarray, random_state=74648, test_size=TEST_SIZE)

    TRAIN_BATCH_SIZE = train_size // 20
    TEST_BATCH_SIZE = 1

    NUM_LAYERS = 1  # len(train_x[0])

    clf = RandomForestClassifier(max_depth=3, random_state=0)

    clf.fit(X_train, t_train)

    pred = clf.predict(X_train)
    acc = [p == t_train[i] for i, p in enumerate(pred)].count(True)/len(pred)
    cm = confusion_matrix(pred, t_train)
    sns.heatmap(cm)
    plt.xlabel("true label")
    plt.ylabel("predict")
    plt.savefig('confusion_matrix_randomforest_train.png')
    print(acc)
    print(cm)

    pred = clf.predict(X_test)
    acc = [p == t_test[i] for i, p in enumerate(pred)].count(True)/len(pred)

    cm = confusion_matrix(pred, t_test)
    sns.heatmap(cm)
    plt.xlabel("true label")
    plt.ylabel("predict")
    plt.savefig('confusion_matrix_randomforest.png')

    print(acc)
    print(cm)
    exit()


if __name__ == "__main__":
    train()