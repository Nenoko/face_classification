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

from tqdm import tqdm

import os
import datetime

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

onso_facepoint = []

CLASS_SIZE = 3
test_conf_mat = [[0 for _ in range(CLASS_SIZE)]for _ in range(CLASS_SIZE)]

# 分類器
model = torch.load('./model_main.pth')

kind_of_data_list = ["positive", "neutral", "negative"]
kind_of_data_nums = [i for i in range(len(kind_of_data_list))]

accuracy = []

for kind_of_data, kind_of_data_num in zip(kind_of_data_list, kind_of_data_nums):
    generated_face = torch.load(
        './generated_face_{}.pickle'.format(kind_of_data), map_location=torch.device('cpu'))
    # for seq in zip(generated_face[0]):
    faces_sequence = generated_face[1]
    for faces in tqdm(faces_sequence):
        data = torch.FloatTensor(faces.float())
        output = model(data)

        for o in output:
            pred_label = torch.argmax(o).item()
            # 正解？
            accuracy.append(pred_label == kind_of_data_num)
            test_conf_mat[pred_label][kind_of_data_num] += 1

    print(accuracy.count(True)/len(accuracy))

    plt.clf()  # 初期化
    ziku = ['Positive', 'Neutral', 'Negative']
    df = pd.DataFrame(data=test_conf_mat, index=ziku, columns=ziku)
    sns.heatmap(df, cmap='Blues', annot=True, fmt="d")
    plt.xlabel("true label")
    plt.ylabel("predict")
    plt.savefig('confusion_matrix_NN_generated.png')
