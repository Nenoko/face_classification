import matplotlib
import platform
if platform.system() == 'Darwin':
    print('on Mac ')
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import torch
import numpy as np

import pandas as pd
import seaborn as sns

import tqdm

import os
import datetime

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

onso_facepoint = []

CLASS_SIZE = 3
test_conf_mat = [[0 for _ in range(CLASS_SIZE)]for _ in range(CLASS_SIZE)]

# 分類器
#model = torch.load('./model_main.pth')
clf = torch.load('./model_svm.pth')

kind_of_data_list = ["positive", "neutral", "negative"]
kind_of_data_nums = [i for i in range(len(kind_of_data_list))]
for kind_of_data, kind_of_data_num in zip(kind_of_data_list, kind_of_data_nums):
    generated_face = torch.load(
        './generated_face_{}.pickle'.format(kind_of_data), map_location=torch.device('cpu'))
    # for seq in zip(generated_face[0]):
    seq = generated_face[0]
    data = torch.FloatTensor(torch.stack(seq))
    #output = model(data)
    data_nd = data.to("cpu").detach().numpy().copy()
    data_nd_ = []
    for hatuwa in data_nd:
        data_nd_.append(np.mean(hatuwa, axis=0))
    data = data_nd_

    output = clf.predict(data)
    # 正解率の計算
    batch_acc = [o == kind_of_data_num
                 for i, o in enumerate(output)].count(True) / len(output)

#    accuracy = sum(batch_acc) / len(batch_acc) * 100
#   print('accuracy=%.7f' %
#          (accuracy))

    for i, o in enumerate(output):
        test_conf_mat[o][kind_of_data_num] += 1

    plt.clf()  # 初期化
    ziku = ['Positive', 'Neutral', 'Negative']
    df = pd.DataFrame(data=test_conf_mat, index=ziku, columns=ziku)
    sns.heatmap(df, cmap='Blues', annot=True, fmt="d")
    plt.xlabel("true label")
    plt.ylabel("predict")
    plt.savefig('confusion_matrix_svm_generated.png')
