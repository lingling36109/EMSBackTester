import helper
import torch
from torch import nn


# The regular LSTM class
class LSTM(nn.Module):
    # Define LSTM layer with linear layer at the very end
    def __init__(self, num_features, hidden_units=16, num_layers=1, dropout=0.1):
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
            dropout=dropout,
        )

        self.linear = nn.Linear(in_features=hidden_units, out_features=1)
        self.relu = nn.ReLU()

    # Defines forward propagation for each batch
    def forward(self, inputX):
        batch_size = inputX.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        _, (hn, _) = self.lstm(inputX, (h0, c0))
        out = self.relu(self.linear(hn[-1]).flatten())
        return out

    # Autoregression portion of the LSTM
    # For future reference should maybe have to change for when you put in previous SOH into input
    def predict(self, inputX):
        _, (hn, cn) = self.lstm(inputX, (self.hidden, self.control))
        out = self.sigmoid(self.linear(hn[-1]).flatten())
        self.hidden = hn
        self.control = cn
        return out


# Defined the training loop
if __name__ == "__main__":
    df_train, _, _ = helper.get_dataset("data/battery_log_processed.csv")
    df_train_loader = helper.get_loaders(df_train)

    LSTM_model = LSTM(len(list(df_train.columns.difference(["SOH[%]"]))))
    helper.trainer(df_train_loader, LSTM_model, "loss1.csv")
