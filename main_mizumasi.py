import torch
from torch import nn
from torch.optim import SGD, Adam
# from torch.utils.data.dataset import Subset
import torch.nn.utils.rnn as rnn
# import torch.nn.functional as F
from model import face_classifier
# from params import *
import numpy as np


import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import *

from torch.utils.data import DataLoader, TensorDataset

from torchsummary import summary

from torch.nn.utils.rnn import pad_sequence

import random
from tqdm import tqdm

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

    # generatedと同様の方法で水増しする
    train_x_messed_up = []
    train_t_messed_up = []
    min_wide = 10

#    for x_, t_ in zip(train_x, train_t):
#        for startpoint in range(0, len(x_)-1-min_wide):
#            # 時系列の並びを維持したまま窓で切り抜く
#            for endpoint in range(len(x_)-min_wide, len(x_)):
#
#                tmpx = x_[startpoint:endpoint]
#
#                train_x_messed_up.append(tmpx)
#                train_t_messed_up.append(t_)

    for x_, t_ in tqdm(zip(train_x, train_t)):
        for startpoint in range(0, len(x_) - 1 - min_wide):
            endpoint = startpoint + min_wide
            tmpx = x_[startpoint:endpoint]
            train_x_messed_up.append(tmpx)
            train_t_messed_up.append(t_)

    train_x = train_x_messed_up
    train_t = train_t_messed_up

    train_x = train_x_messed_up
    train_t = train_t_messed_up

    # BATCH_SIZE=1

    # CVのための分割点決め
    n_samples = len(train_x)
    train_size = n_samples * 9 // 10
    test_size = n_samples - train_size

    print("original_train_size = {}".format(train_size))
    print("original_test_size = {}".format(test_size))

    # ds = TensorDataset(train_x, train_t)
    ds = list(zip(train_x, train_t))
    # 混ぜる
    random.shuffle(ds)

    # split
    ds_train = ds[:train_size]
    ds_test = ds[train_size:]

    # train_x = train_x[:train_size]
    # train_t = train_t[:train_size]

    TRAIN_BATCH_SIZE = train_size // 30
    # TEST_BATCH_SIZE = test_size // 20
    # TRAIN_BATCH_SIZE = 1
    TEST_BATCH_SIZE = 1
    print("train_batch_size = {}".format(TRAIN_BATCH_SIZE))

    # loader_train = DataLoader(
    #    ds_train, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    # loader_test = DataLoader(
    #    ds_test, batch_size=TEST_BATCH_SIZE, shuffle=False)

    # input_size  ... 68次元ランドマーク法で取得された顔表情点
    # hidden_size ... 隠れ層サイズ
    # num_layers  ... レイヤー数　今回は表情点列（可変長）の最大長になる
    NUM_LAYERS = 1  # len(train_x[0])
    # input_size, hidden_size, num_layers, class_size):
    HIDDEN_SIZE = 100
    CLASS_SIZE = 3
    EPOCHS_NUM = 20
    model = face_classifier(
        68 * 2, HIDDEN_SIZE, NUM_LAYERS, CLASS_SIZE)  # modelの宣言
    # summary(model, input_size=(25, 68*2))

    criterion = nn.CrossEntropyLoss()
    lr = 0.01
    # lr = 0.1
    # optimizer = Adam(model.parameters(), lr=lr)  # 最適化関数の宣言
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)  # 最適化関数の宣言

    # debug
    training_loss_list = []
    test_loss_list = []

    training_accuracy_list = []
    test_accuracy_list = []

    total_acc = []
    train_conf_mat = [[0 for _ in range(CLASS_SIZE)]for _ in range(CLASS_SIZE)]
    test_conf_mat = [[0 for _ in range(CLASS_SIZE)]for _ in range(CLASS_SIZE)]

    for epoch in range(EPOCHS_NUM):
        # training
        model.train()
        running_loss = 0.0
        random.shuffle(ds_train)
        print("EPOCH : {}".format(epoch))
        epoch_accuracy = []
        print("training ... ")
        for data, label in tqdm(ds_train):

            # if len(data) < TRAIN_BATCH_SIZE:
            #    break

            optimizer.zero_grad()

            # data = torch.FloatTensor(data)
            # label = torch.LongTensor(label)

            output = model(data.unsqueeze(0))

            loss = criterion(output.float(), label.unsqueeze(0))
            loss.backward()
            optimizer.step()

            running_loss += loss.data.item()
            # 正解率の計算
            epoch_accuracy.append(torch.argmax(output) == label.item())

        accuracy = epoch_accuracy.count(True) / len(epoch_accuracy) * 100
        if epoch % PRINT_EPOCH == 0:

            print('%d training loss: %.7f , accuracy=%.7f' %
                  (epoch, running_loss, accuracy))
        training_loss_list.append(running_loss)
        training_accuracy_list.append(accuracy)
        print("test ... ")
        # test
        running_loss = 0.0
        epoch_accuracy = []
        model.eval()
        for data, label in tqdm(ds_test):

            # if len(data) < TRAIN_BATCH_SIZE:
            #    break

            # optimizer.zero_grad()

            # data = torch.FloatTensor(data)
            # label = torch.LongTensor(label)

            output = model(data.unsqueeze(0))

            loss = criterion(output.float(), label.unsqueeze(0))
            # loss.backward()
            # optimizer.step()

            running_loss += loss.data.item()
            # 正解率の計算
            epoch_accuracy.append(torch.argmax(output) == label.item())

        accuracy = epoch_accuracy.count(True) / len(epoch_accuracy) * 100
        if epoch % PRINT_EPOCH == 0:

            print('%d test loss: %.7f , accuracy=%.7f' %
                  (epoch, running_loss, accuracy))
        test_loss_list.append(running_loss)
        test_accuracy_list.append(accuracy)

    train_conf_mat = [[0 for _ in range(CLASS_SIZE)]
                      for _ in range(CLASS_SIZE)]

    for data, label in ds_train:

        #        if len(data) < TRAIN_BATCH_SIZE:
        #            break

        #data = torch.FloatTensor(data)
        #label = torch.LongTensor(label)
        output = model(data)

        o = torch.argmax(output).item()
        train_conf_mat[o][label.item()] += 1

    test_conf_mat = [
        [0 for _ in range(CLASS_SIZE)]for _ in range(CLASS_SIZE)]
    for data, label in ds_test:

        # if len(data) < TEST_BATCH_SIZE:
        #    break

        #data = torch.FloatTensor(data)
        #label = torch.LongTensor(label)
        output = model(data)

        # for i, o_ in enumerate(output):
        test_conf_mat[o][label.item()] += 1

    with open('./training_loss.pkl', 'wb') as f:
        torch.save(training_loss_list, f)
    with open('./test_loss.pkl', 'wb') as f:
        torch.save(test_loss_list, f)
    with open('./training_accuracy.pkl', 'wb') as f:
        torch.save(training_accuracy_list, f)
    with open('./test_accuracy.pkl', 'wb') as f:
        torch.save(test_accuracy_list, f)
    with open('./model_main.pth', 'wb') as f:
        torch.save(model, f)

    draw(training_loss_list, "training_loss", "loss")
    draw(test_loss_list, "test_loss", "loss")
    draw(training_accuracy_list, "training_accuracy", "accuracy[%]")
    draw(test_accuracy_list, "test_accuracy", "accuracy[%]")

    print(train_conf_mat)

    plt.clf()  # 初期化
    ziku = ['Positive', 'Neutral', 'Negative']
    df = pd.DataFrame(data=train_conf_mat, index=ziku, columns=ziku)
    sns.heatmap(df, cmap='Blues', annot=True, fmt="d")
    plt.xlabel("true label")
    plt.ylabel("predict")
    plt.savefig('confusion_matrix_NN_train.png')

    print(test_conf_mat)

    plt.clf()  # 初期化
    ziku = ['Positive', 'Neutral', 'Negative']
    df = pd.DataFrame(data=test_conf_mat, index=ziku, columns=ziku)
    sns.heatmap(df, cmap='OrRd', annot=True, fmt="d")
    plt.xlabel("true label")
    plt.ylabel("predict")
    plt.savefig('confusion_matrix_NN.png')


def draw(data, title, ylabel):
    xlabel = 'epoch'
    label = [(i+1) for i, _ in enumerate(data)]

    plt.plot(label, data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.gca().get_yaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    if "accuracy" in title:
        plt.ylim = (min(data) - 0.5, max(data) + 0.5)
    plt.savefig("{}.png".format(title))
    plt.gca().clear()


# confusion matrixを出す
# loader_all = DataLoader(
#    ds, shuffle=True)
# model.eval()


if __name__ == "__main__":
    train()
