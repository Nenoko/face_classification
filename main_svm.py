# from params import *
import numpy as np

import torch

import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import seaborn as sns
from torchsummary import summary

PRINT_EPOCH = 2


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
    #mindist = 5
    #minseed = 6900
    #minzero = 15
    #minone = 20
    #mintwo = 20
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
        # train_x_ndarray, train_t_ndarray, random_state=74648, test_size=TEST_SIZE)
        train_x_ndarray, train_t_ndarray,  test_size=TEST_SIZE)

    TRAIN_BATCH_SIZE = train_size // 20
    TEST_BATCH_SIZE = 1

    NUM_LAYERS = 1  # len(train_x[0])

    #clf = SVC(kernel='linear', random_state=None, verbose=2)
    clf = SVC(kernel='rbf', random_state=None, verbose=2)

    clf.fit(X_train, t_train)

    pred = clf.predict(X_train)
    acc = [p == t_train[i] for i, p in enumerate(pred)].count(True)/len(pred)
    cm = confusion_matrix(pred, t_train)
    ziku = ['Positive', 'Neutral', 'Negative']
    df = pd.DataFrame(data=cm, index=ziku, columns=ziku)
    sns.heatmap(df, cmap='Blues', annot=True, fmt="d")
    plt.xlabel("true label")
    plt.ylabel("predict")
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.gca().get_yaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.savefig('confusion_matrix_svm_train.png')
    print(acc)
    print(cm)

    pred = clf.predict(X_test)
    acc = [p == t_test[i] for i, p in enumerate(pred)].count(True)/len(pred)

    cm = confusion_matrix(pred, t_test)
    ziku = ['Positive', 'Neutral', 'Negative']
    df = pd.DataFrame(data=cm, index=ziku, columns=ziku)
    sns.heatmap(df, cmap='OrRd', annot=True, fmt="d")
    plt.xlabel("true label")
    plt.ylabel("predict")
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.gca().get_yaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.savefig('confusion_matrix_svm.png')
    print("")
    print(acc)
    print(cm)
    with open('./model_svm.pth', 'wb') as f:
        torch.save(clf, f)


if __name__ == "__main__":
    train()
