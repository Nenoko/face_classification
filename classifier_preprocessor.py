import torch
import re
from torch import nn
import torch.nn.utils.rnn as rnn
from params import TARGET_CLASS
import itertools

# 顔→ネガ/ポジ/無のペアデータセットを作成


def preprocess_one_label(dataset_path, label):
    onso_facepoint = []

    # onso_facepoint=pickle.load(f)
    onso_facepoint = torch.load(
        dataset_path, map_location=torch.device("cuda:0"))
    # debug
    # onso_facepoint=torch.load('./super_positive_onso_facepoint.pickle',map_location=torch.device('cuda:0'))
    x_list = []
    y_list = []
    for sentence in onso_facepoint:
        x_list_ = []
        y_list_ = []
        for word in sentence:
            x_list__ = []
            y_list__ = []
            for onso, facepoint in word:
                x_list__.append(onso)
                y_list__.append(facepoint)
            x_list_.append(x_list__)
            y_list_.append(y_list__)
        if len(x_list_) == 0 or len(y_list_) == 0:
            continue
        x_list.append(x_list_)
        y_list.append(y_list_)

    x_list = y_list
#    # 音素を前処理
#    # アクセント文字を削除
#    preprocessed_trainx = []
#    for sentence in x_list:
#        preprocessed_trainx_ = []
#        for word in sentence:
#            preprocessed_trainx__ = []
#            for onso in word:
#                no_accent_onso = re.sub("\d+", "", onso)
#                preprocessed_trainx__.append(no_accent_onso)
#                preprocessed_trainx_.append(preprocessed_trainx__)
#        preprocessed_trainx.append(preprocessed_trainx_)
#
#    trainx_index = []
#    for sentence in preprocessed_trainx:
#        sentence_index = []
#        for word in sentence:
#            sentence_index.extend(sentence2index(word))
#            # sentence_index.append(sentence2index(word))
#        trainx_index.append(torch.tensor(sentence_index, dtype=torch.long))
#    preprocessed_trainx = rnn.pad_sequence(
#        trainx_index, batch_first=True, padding_value=VOCAB_SIZE
#    )
#    x_list = preprocessed_trainx
#    SENTENCE_SIZE = len(x_list[0])
#
    # 今回はgivenなデータ=表情点となる
    preprocessed_trainx = []
    for face_series_in_sentence in x_list:
        face_series = []
        for face_series_in_word in face_series_in_sentence:
            for face in face_series_in_word:
                face_series.append(
                    list(itertools.chain.from_iterable(face)))
        preprocessed_trainx.append(
            torch.tensor(face_series, dtype=torch.long)
        )

    preprocessed_trainx = rnn.pad_sequence(
        preprocessed_trainx, batch_first=True)

    x_list = preprocessed_trainx
    y_list = [label for _ in x_list]

    return x_list, y_list


def preprocess():
    x_list = []
    y_list = []
    x_list_, y_list_ = preprocess_one_label(
        "./positive_onso_facepoint.pkl", TARGET_CLASS.POSITIVE.value
    )
    x_list.extend(x_list_)
    y_list.extend(y_list_)
    x_list_, y_list_ = preprocess_one_label(
        "./negative_onso_facepoint.pkl", TARGET_CLASS.NEGATIVE.value
    )
    x_list.extend(x_list_)
    y_list.extend(y_list_)
    x_list_, y_list_ = preprocess_one_label(
        "./neutral_onso_facepoint.pkl", TARGET_CLASS.NEUTRAL.value
    )
    x_list.extend(x_list_)
    y_list.extend(y_list_)

    x_list = rnn.pad_sequence(x_list, batch_first=True, padding_value=0)

    y_list = torch.tensor(y_list, dtype=torch.long)

    with open("./classifier_train_data.pkl", "wb") as f:
        torch.save([x_list, y_list], f)


if __name__ == '__main__':
    preprocess()
