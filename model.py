from torch import nn
import torch.nn.functional as F


class face_classifier(nn.Module):
    # input_size  ... 68次元ランドマーク法で取得された顔表情点
    # hidden_size ... 隠れ層サイズ
    # num_layers  ... レイヤー数　今回は表情点列（可変長）の最大長になる
    def __init__(self, input_size, hidden_size, num_layers=1, class_size=3):

        # 親クラスのコンストラクタ。決まり文句
        super(face_classifier, self).__init__()
        #face_stream_length = 25
        # models
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        #self.hidden = nn.Linear(input_size * face_stream_length, hidden_size)
        self.dense = nn.Linear(hidden_size, class_size)

    def forward(self, input):
        hidden_output, _ = self.lstm(input)
        hidden_output = hidden_output[:, -1, :]
        #hidden_output = self.hidden(input.view(len(input), -1))
        dense_output = self.dense(hidden_output)

        # return dense_output
        return F.softmax(dense_output, dim=1)


# print(face_classifier)
