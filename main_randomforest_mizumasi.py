import numpy as np
import torch

import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from tqdm import tqdm

PRINT_EPOCH = 2


def train():
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_x, train_t = torch.load('./classifier_train_data.pkl')

    # generatedと同様の方法で水増しする
    train_x_messed_up = []
    train_t_messed_up = []
    min_wide = 10

    for x_, t_ in tqdm(zip(train_x, train_t)):
        for startpoint in range(0, len(x_) - 1 - min_wide):
            endpoint = startpoint + min_wide
            tmpx = x_[startpoint:endpoint]
            # train_x_messed_up.append(tmpx)
            train_x_messed_up.append(tmpx[:, 16 * 2:])
            train_t_messed_up.append(t_)

    train_x = train_x_messed_up
    train_t = train_t_messed_up

    # BATCH_SIZE=1

    # CVのための分割点決め
    n_samples = len(train_x)
    train_size = n_samples * 9 // 10
    # test_size = n_samples - train_size

    train_x_ndarray = torch.stack(train_x).to('cpu').detach().numpy().copy()
    train_x_ndarray_ = []
    for hatuwa in train_x_ndarray:
        train_x_ndarray_.append(np.mean(hatuwa, axis=0))
    train_x_ndarray = train_x_ndarray_

    train_t_ndarray = torch.stack(train_t).to('cpu').detach().numpy().copy()

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
        # train_x_ndarray, train_t_ndarray, random_state=74648, test_size=TEST_SIZE)
        train_x_ndarray, train_t_ndarray,  test_size=TEST_SIZE)

    TRAIN_BATCH_SIZE = train_size // 20
    TEST_BATCH_SIZE = 1

    NUM_LAYERS = 1  # len(train_x[0])

    clf = RandomForestClassifier(max_depth=5, random_state=0)

    clf.fit(X_train, t_train)

    # Feature Importance
    fti = clf.feature_importances_
#   feature_dict = dict(zip(["{}x".format(
#       i // 2) if i % 2 == 0 else "{}y".format(i // 2 + 1) for i in range(len(fti))], fti))
#   feature_dict = list(
#       reversed(sorted(feature_dict.items(), key=lambda x: x[1])))
#
#   print('Feature Importances:')
#    for i, feat in (feature_dict):
#        # 表示限界
#        if feat < 0.1**6:
#            break
#        print('\t{0:20s} : {1:>.6f}'.format(i, feat))

    fti_sum = [fti[2 * i] + fti[2 * i + 1]for i in range(len(fti) // 2)]

    feature_dict = dict(zip([i for i in range(len(fti))], fti_sum))
    feature_dict = list(
        reversed(sorted(feature_dict.items(), key=lambda x: x[1])))

    print('Feature Importances:')
    for i, feat in (feature_dict):
        # 表示限界
        if feat < 0.1**6:
            break
        print('\t{0:20d} : {1:>.6f}'.format(i, feat))

    # 棒グラフを描く
    feature_dict_key = [str(int(key)) for key, val in feature_dict]
    feature_dict_val = [val * 100 for key, val in feature_dict]

    display_limit = 0
    tmp_sum = 0
    for j, val in enumerate(feature_dict_val):
        tmp_sum += val
        if j == 15:
            print(tmp_sum)
        if tmp_sum > 99:
            display_limit = j
            break
    print(display_limit)
    feature_dict_key = feature_dict_key[:display_limit]
    feature_dict_val = feature_dict_val[:display_limit]

    with open("feature_dict.pkl", "wb") as f:
        torch.save([feature_dict_key, feature_dict_val], f)

    plt.clf()
    #plt.bar(feature_dict_key, feature_dict_val)
    plt.bar(feature_dict_key[:15], feature_dict_val[:15], color='orange')
    plt.show()

    plt.clf()
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
    plt.savefig('confusion_matrix_randomforest_train.png')
    print(acc)
    print(cm)

    cm = None

    plt.clf()
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
    plt.savefig('confusion_matrix_randomforest.png')

    print(acc)
    print(cm)

    with open('./model_randomforest.pth', 'wb') as f:
        torch.save(clf, f)


if __name__ == "__main__":
    train()
