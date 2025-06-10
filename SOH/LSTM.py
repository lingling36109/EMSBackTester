from tkinter import Variable

import helper
import torch
from torch import nn


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
        self.sigmoid = nn.Sigmoid()

    # Defines hidden states for prediction phase
    def init_hidden(self, batch_size):
        self.hidden = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        self.control = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

    # Defines forward propagation for each batch
    def forward(self, inputX):
        batch_size = inputX.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        _, (hn, _) = self.lstm(inputX, (h0, c0))
        out = self.sigmoid(self.linear(hn[-1]).flatten())
        return out

    # Autoregression portion of the LSTM
    # For future reference should maybe have to change for when you put in previous SOH into input
    def predict(self, inputX):
        _, (hn, cn) = self.lstm(inputX, (self.hidden, self.control))
        out = self.sigmoid(self.linear(hn[-1]).flatten())
        self.hidden = hn
        self.control = cn
        return out


# Built test function for the LSTM
def test_LSTM(data_loader, model: LSTM, loss_function):
    f = open("loss1B.csv", "w")
    num_batch = len(data_loader)
    total_loss = 0

    model.init_hidden(batch_size=num_batch)

    with torch.no_grad():
        for X, y in data_loader:
            output = model.predict(X)
            loss = loss_function(output, y)
            total_loss += loss.item()

    avg_loss = total_loss / num_batch
    print(f"LSTM Test loss: {avg_loss}")
    f.write(avg_loss.__str__() + ",")
    return avg_loss


# Defined the training loop
if __name__ == "__main__":
    df_train, _, _ = helper.get_dataset("data/battery_log_processed.csv")
    df_train_loader = helper.get_loaders(df_train)

    LSTM_model = LSTM(len(list(df_train.columns.difference(["SOH[%]"]))))
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(LSTM_model.parameters(), lr=0.005)

    f = open("loss1A.csv", "w")
    print(" -- Starting classical loss -- ")
    for ix_epoch in range(10):
        print(f"Epoch {ix_epoch}\n---------")

        total_loss = 0
        num_batch = len(df_train_loader)
        for X, y in df_train_loader:
            output = LSTM_model.forward(X)
            loss = loss_function(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batch
        f.write(avg_loss.__str__() + ",")
        print(f"LSTM Train loss per batch: {avg_loss}")
