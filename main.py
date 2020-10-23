import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.utils.data.dataset import Subset
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
from model import face_classifier
from params import *
import numpy as np


import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from torch.utils.data import DataLoader, TensorDataset

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
    train_size = n_samples*8//10
    test_size = n_samples - train_size

    ds = TensorDataset(train_x, train_t)
    ds_train, ds_test = torch.utils.data.random_split(
        ds, [train_size, test_size])

    TRAIN_BATCH_SIZE = train_size // 30
    TEST_BATCH_SIZE = test_size // 30
    # TRAIN_BATCH_SIZE = 1
    # TEST_BATCH_SIZE = 1

    loader_train = DataLoader(
        ds_train, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    loader_test = DataLoader(
        ds_test,  batch_size=TEST_BATCH_SIZE, shuffle=False)

    packed = rnn.pad_sequence(train_x, batch_first=True)

    # input_size  ... 68次元ランドマーク法で取得された顔表情点
    # hidden_size ... 隠れ層サイズ
    # num_layers  ... レイヤー数　今回は表情点列（可変長）の最大長になる
    NUM_LAYERS = len(train_x[0])
    # input_size, hidden_size, num_layers, class_size):
    # NUM_LAYERS = 50
    # HIDDEN_SIZE = 100
    # CLASS_SIZE = 3
    # EPOCHS_NUM = 20
    model = face_classifier(
        68*2, HIDDEN_SIZE, NUM_LAYERS, CLASS_SIZE)  # modelの宣言

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)  # 最適化関数の宣言

    # debug
    training_loss_list = []
    test_loss_list = []

    training_accuracy_list = []
    test_accuracy_list = []
    
    total_acc = []
    for epoch in range(EPOCHS_NUM):
        # training
        model.train()
        running_loss = 0.0
        for data, label in loader_train:

            if len(data) < TRAIN_BATCH_SIZE:
                break

            optimizer.zero_grad()

            data = torch.LongTensor(data).float()
            label = torch.LongTensor(label)

            output = model(data)

            loss = criterion(output.float(), label)
            loss.backward()
            optimizer.step()

            # 正解率の計算
            batch_accuracy = [torch.argmax(o).item() == label[i]
                              for i, o in enumerate(output)].count(True) / len(label)
            total_acc.append(batch_accuracy)
            running_loss += loss.data.item()

        accuracy = sum(total_acc)/len(total_acc) * 100
        if epoch % PRINT_EPOCH == 0:

            print('%d training loss: %.3f , accuracy=%.3f' %
                  (epoch, running_loss, accuracy))
        training_loss_list.append(running_loss)
        training_accuracy_list.append(accuracy)
        # test
        running_loss = 0.0
        total_acc = []
        model.eval()
        for data, label in loader_test:

            if len(data) < TEST_BATCH_SIZE:
                break

            data = torch.LongTensor(data).float()
            label = torch.LongTensor(label)

            output = model(data)

            loss = criterion(output.float(), label)
            # loss.backward()
            # optimizer.step()

            running_loss += loss.data.item()
            # 正解率の計算
            batch_accuracy = [torch.argmax(o).item() == label[i]
                              for i, o in enumerate(output)].count(True) / len(label)
            total_acc.append(batch_accuracy)

        accuracy = sum(total_acc)/len(total_acc) * 100
        if epoch % PRINT_EPOCH == 0:
            print('%d test loss: %.3f , accuracy=%.3f' %
                  (epoch, running_loss, accuracy))
        test_loss_list.append(running_loss)
        test_accuracy_list.append(running_loss)


    print(training_loss_list)
    print(test_loss_list)
    with open('./training_loss.pkl', 'wb') as f:
        torch.save(training_loss_list, f)
    with open('./test_loss.pkl', 'wb') as f:
        torch.save(test_loss_list, f)
    with open('./training_accuracy.pkl', 'wb') as f:
        torch.save(training_accuracy_list, f)
    with open('./test_accuracy.pkl', 'wb') as f:
        torch.save(test_accuracy_list, f)

    # confusion matrixを出す
    loader_all = DataLoader(
        ds, shuffle=True)
    model.eval()
#    for data, label in loader_all:
#        data = torch.LongTensor(data).float()
#        label = torch.LongTensor(label)
#
#        output = model(data)
#        # 確率最大のインデックスを取る
#        output_argmax = torch.argmax(output, dim=1)
#
#    # 両者を2次元プロット
#    print_cmx(label, output_argmax)

    # ロスもplot
    left = [i for i in range(len(training_loss_list))]
    height = np.array(training_loss_list)
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.plot(left, height, label='train')
    plt.show()
    left = [i for i in range(len(test_loss_list))]
    height = np.array(test_loss_list)
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.plot(left, height, label='test')
    plt.show()


if __name__ == "__main__":
    train()
