import sys
import torch
from torch import nn
from torch.optim import SGD, Adam
from model import face_classifier

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import random
from tqdm import tqdm

PRINT_EPOCH = 2


def train():
    train_x, train_t = torch.load('./classifier_train_data.pkl')
    # train_x, train_t = torch.load('./classifier_train_data_mizumasi.pkl')

    # BATCH_SIZE=1
    # CVのための分割点決め
    n_samples = len(train_x)
    train_size = n_samples * 9 // 10

    ds = list(zip(train_x, train_t))
    # 混ぜる
    random.shuffle(ds)

    # split
    ds_train = ds[:train_size]
    ds_test = ds[train_size:]

    TRAIN_BATCH_SIZE = train_size // 30
    print("train_batch_size = {}".format(TRAIN_BATCH_SIZE))

    # input_size  ... 68次元ランドマーク法で取得された顔表情点
    # hidden_size ... 隠れ層サイズ
    # num_layers  ... レイヤー数　今回は表情点列（可変長）の最大長になる
    NUM_LAYERS = 1  # len(train_x[0])
    HIDDEN_SIZE = 100
    CLASS_SIZE = 3
    EPOCHS_NUM = 20
    # input_size, hidden_size, num_layers, class_size):
    model = face_classifier(
        68 * 2, HIDDEN_SIZE, NUM_LAYERS, CLASS_SIZE)  # modelの宣言

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

    for epoch in range(EPOCHS_NUM):
        # training
        model.train()
        training_loss = 0.0
        random.shuffle(ds_train)
        epoch_accuracy = []
        with tqdm(ds_test, leave=False) as ds_train_tqdm:
            ds_train_tqdm.set_description('EPOCH {}'.format(epoch))
            for data, label in ds_train_tqdm:
                optimizer.zero_grad()
                output = model(data.unsqueeze(0))

                loss = criterion(output.float(), label.unsqueeze(0))
                loss.backward()
                optimizer.step()

                training_loss += loss.data.item()
                # 正解率の計算
                epoch_accuracy.append(torch.argmax(output) == label.item())

        training_accuracy = epoch_accuracy.count(
            True) / len(epoch_accuracy) * 100
        training_loss_list.append(training_loss)
        training_accuracy_list.append(training_accuracy)

        # test
        test_loss = 0.0
        epoch_accuracy = []
        model.eval()
        with tqdm(ds_test, leave=False) as ds_test_tqdm:
            ds_test_tqdm.set_description('EPOCH {}'.format(epoch))
            for data, label in ds_test_tqdm:
                output = model(data.unsqueeze(0))
                # 損失,正解率の計算
                test_loss += loss.data.item()
                epoch_accuracy.append(torch.argmax(output) == label.item())

            test_accuracy = epoch_accuracy.count(
                True) / len(epoch_accuracy) * 100
            test_loss_list.append(test_loss)
            test_accuracy_list.append(test_accuracy)

            if epoch % PRINT_EPOCH == 0:
                print('%d training loss: %.7f , accuracy=%.7f' %
                      (epoch, training_loss, training_accuracy))
                print('%d test loss: %.7f , accuracy=%.7f' %
                      (epoch, test_loss, test_accuracy))

    train_conf_mat = [[0 for _ in range(CLASS_SIZE)]
                      for _ in range(CLASS_SIZE)]
    for data, label in ds_train:
        output = model(data.unsqueeze(0))
        o = torch.argmax(output).item()
        train_conf_mat[o][label.item()] += 1

    test_conf_mat = [
        [0 for _ in range(CLASS_SIZE)]for _ in range(CLASS_SIZE)]
    for data, label in ds_test:
        output = model(data.unsqueeze(0))
        test_conf_mat[o][label.item()] += 1

    with open('loss/training.pkl', 'wb') as f:
        torch.save(training_loss_list, f)
    with open('loss/test.pkl', 'wb') as f:
        torch.save(test_loss_list, f)
    with open('accuracy/training.pkl', 'wb') as f:
        torch.save(training_accuracy_list, f)
    with open('accuracy/test.pkl', 'wb') as f:
        torch.save(test_accuracy_list, f)
    with open('model/main.pth', 'wb') as f:
        torch.save(model, f)

    draw(training_loss_list, "training_loss", "loss")
    draw(test_loss_list, "test_loss", "loss")
    draw(training_accuracy_list, "training_accuracy", "accuracy[%]")
    draw(test_accuracy_list, "test_accuracy", "accuracy[%]")

    # plt.clf()  # 初期化
    # df = pd.DataFrame(data=train_conf_mat, index=ziku, columns=ziku)
    # sns.heatmap(df, cmap='Blues', annot=True, fmt="d")
    # plt.xlabel("true label")
    # plt.ylabel("predict")
    # plt.savefig('confusion_matrix_NN_train.png')
    print(train_conf_mat)
    ziku = ['Positive', 'Neutral', 'Negative']
    draw_cm(train_conf_mat, ziku, 'Blues', 'confusion_matrix/NN/train.png')

    # plt.clf()  # 初期化
    # ziku = ['Positive', 'Neutral', 'Negative']
    # df = pd.DataFrame(data=test_conf_mat, index=ziku, columns=ziku)
    # sns.heatmap(df, cmap='OrRd', annot=True, fmt="d")
    # plt.xlabel("true label")
    # plt.ylabel("predict")
    # plt.savefig('confusion_matrix_NN.png')
    print(test_conf_mat)
    draw_cm(test_conf_mat, ziku, 'Blues', 'confusion_matrix/NN/test.png')


def draw(data, title, ylabel):
    xlabel = 'epoch'
    label = [(i + 1) for i, _ in enumerate(data)]

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


def draw_cm(cm, ziku, color, filename):
    plt.clf()  # 初期化
    # ziku = ['Positive', 'Neutral', 'Negative']
    df = pd.DataFrame(data=cm, index=ziku, columns=ziku)
    sns.heatmap(df, cmap=color, annot=True, fmt="d")
    plt.xlabel("true")
    plt.ylabel("predict")
    plt.savefig(filename)
    plt.clf()  # 初期化


if __name__ == "__main__":
    train()
