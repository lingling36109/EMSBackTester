import helper
import torch
from torch import nn


# LSTM + CNN class, CNN used for feature extraction
class CNN_LSTM(helper.base):
    def __init__(self, num_features, conv_filters=64, kernel_size=3, hidden_units=16, num_layers=1):
        super().__init__()
        self.hidden = None
        self.control = None
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

        self.linear = nn.Linear(in_features=hidden_units, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, inputX):
        batch_size = inputX.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        inputX = inputX.permute(0, 2, 1)
        inputX = self.cnn(inputX)
        inputX = inputX.permute(0, 2, 1)

        _, (hn, _) = self.lstm(inputX, (h0, c0))
        out = self.relu(self.linear(hn[-1]).flatten())
        return out

    def predict(self, inputX):
        inputX = inputX.permute(0, 2, 1)
        inputX = self.cnn(inputX)
        inputX = inputX.permute(0, 2, 1)
        _, (hn, cn) = self.lstm(inputX, (self.hidden, self.control))
        out = self.relu(self.linear(hn[-1]).flatten())
        self.hidden = hn
        self.control = cn
        return out