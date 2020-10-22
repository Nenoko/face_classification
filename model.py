from torch import nn
import torch.nn.functional as F


class face_classifier(nn.Module):
    # input_size  ... 68次元ランドマーク法で取得された顔表情点
    # hidden_size ... 隠れ層サイズ
    # num_layers  ... レイヤー数　今回は表情点列（可変長）の最大長になる
    def __init__(self, input_size, hidden_size, num_layers, class_size):

        # 親クラスのコンストラクタ。決まり文句
        super(face_classifier, self).__init__()
        # models
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.dense = nn.Linear(hidden_size, class_size)

    def forward(self, input):
        hidden_output, _ = self.lstm(input)
        hidden_output = hidden_output[:, -1, :]
        dense_output = self.dense(hidden_output)

        return F.log_softmax(dense_output, dim=1)
