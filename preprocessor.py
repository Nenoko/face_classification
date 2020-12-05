import torch
import torch.nn.utils.rnn as rnn
from params import TARGET_CLASS
import itertools
from tqdm import tqdm

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

    # 平均顔の算出
    x_heikinface = x_list[0][0] / float(len(x_list[0]))
    for i, sentence in enumerate(x_list):
        for j, face in enumerate(sentence):
            if i == 0 and j == 0:
                continue
            x_heikinface += face / float(len(sentence))
    x_heikinface = x_heikinface.float()
    x_heikinface /= len(x_list)

    # 最後に表情点を平均顔からの差分に置き換える
    x_preprocessing = []
    for sentence in x_list:
        x_preprocessing_ = []
        for face in sentence:
            x_preprocessing_.append(torch.tensor(x_heikinface - face))
        x_preprocessing.append(torch.stack(x_preprocessing_))
    x_list = torch.stack(x_preprocessing)
    y_list = [label for _ in x_list]

    return x_list, y_list


def preprocess():
    x_list = []
    y_list = []
    x_list_, y_list_ = preprocess_one_label(
        "./positive_onso_facepoint.pkl", TARGET_CLASS.POSITIVE.value
    )
    print("positive_face_size : {}".format(len(x_list_)))
    x_list.extend(x_list_)
    y_list.extend(y_list_)

    x_list_, y_list_ = preprocess_one_label(
        "./neutral_onso_facepoint.pkl", TARGET_CLASS.NEUTRAL.value
    )
    print("neutral_face_size : {}".format(len(x_list_)))
    x_list.extend(x_list_)
    y_list.extend(y_list_)

    x_list_, y_list_ = preprocess_one_label(
        "./negative_onso_facepoint.pkl", TARGET_CLASS.NEGATIVE.value
    )
    print("negative_face_size : {}".format(len(x_list_)))
    x_list.extend(x_list_)
    y_list.extend(y_list_)
    x_list = rnn.pad_sequence(x_list, batch_first=True, padding_value=0)
    y_list = torch.tensor(y_list, dtype=torch.long)

    with open("./classifier_train_data.pkl", "wb") as f:
        torch.save([x_list, y_list], f)

    # generatedと同様の方法で水増しする
    train_x_messed_up = []
    train_t_messed_up = []
    min_wide = 10

    for x_, t_ in tqdm(zip(x_list, y_list)):
        for startpoint in range(0, len(x_) - 1 - min_wide):
            endpoint = startpoint + min_wide
            tmpx = x_[startpoint:endpoint]
            train_x_messed_up.append(tmpx)
            train_t_messed_up.append(t_)

    x_list = train_x_messed_up
    y_list = train_t_messed_up

    with open("./classifier_train_data_mizumasi.pkl", "wb") as f:
        torch.save([x_list, y_list], f)


if __name__ == '__main__':
    preprocess()
