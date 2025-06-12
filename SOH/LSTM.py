import helper
import torch
from torch import nn


# LSTM class
class LSTM(nn.Module):
    def __init__(self, num_features, hidden_units=16, num_layers=1):
        super().__init__()
        self.hidden = None
        self.control = None
        self.hidden_units = hidden_units
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=num_features,
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

        _, (hn, _) = self.lstm(inputX, (h0, c0))
        out = self.relu(self.linear(hn[-1]).flatten())
        return out

    def predict(self, inputX):
        _, (hn, cn) = self.lstm(inputX, (self.hidden, self.control))
        out = self.relu(self.linear(hn[-1]).flatten())
        self.hidden = hn
        self.control = cn
        return out


# Defined the training loop
if __name__ == "__main__":
    torch.manual_seed(1)
    df_train, _, _ = helper.get_dataset("data/training/battery_log_processed.csv")
    df_train_loader = helper.get_loaders(df_train)

    LSTM_model = LSTM(len(list(df_train.columns.difference(["SOH[%]"]))))
    helper.trainer(df_train_loader, LSTM_model, "data/losses/loss1.csv")
