import helper
import torch
from torch import nn


# The CNN + LSTM + Attention class
class CNN_LSTM_Attention(helper.base):
    def __init__(self, num_features, conv_filters=64, kernel_size=3, hidden_units=16, num_layers=1):
        super().__init__()
        self.hidden_units = hidden_units
        self.num_layers = num_layers

        self.cnn = nn.Conv1d(
            in_channels=num_features,
            out_channels=conv_filters,
            kernel_size=kernel_size,
            padding="same"
        )

        self.lstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=num_layers,
        )

        self.W_1 = nn.Linear(hidden_units, hidden_units)
        self.W_2 = nn.Linear(hidden_units, hidden_units)
        self.v = nn.Linear(hidden_units, 1)

        self.linear = nn.Linear(in_features=hidden_units, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, inputX):
        device = inputX.device
        batch_size = inputX.shape[0]

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).to(device)

        x = inputX.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)

        lstm_out, (hn, _) = self.lstm(x, (h0, c0))

        query = hn[-1]
        query = query.unsqueeze(1)

        score = self.v(torch.tanh(self.W_1(lstm_out) + self.W_2(query)))
        attn_weights = torch.softmax(score, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        out = self.relu(self.linear(context))     # [batch
        return out.squeeze(1)


    def predict(self, inputX):
        raise NotImplementedError


# TFT class
class TFT(helper.base):
    def __init__(self):
        raise NotImplementedError

    def forward(self, inputX):
        raise NotImplementedError

    def predict(self, inputX):
        raise NotImplementedError


# CNN TFT class
class CNN_TFT(helper.base):
    def __init__(self):
        raise NotImplementedError

    def forward(self, inputX):
        raise NotImplementedError

    def predict(self, inputX):
        raise NotImplementedError