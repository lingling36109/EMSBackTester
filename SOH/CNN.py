import helper
import torch
from torch import nn


# LSTM CNN combined, CNN used for feature extraction of the
class LSTM_CNN(helper.base):
    # Constructor for the CNN LSTM
    def __init__(self, num_features, conv_filters, kernel_size, hidden_units=16, num_layers=1, dropout=0.1):
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
            input_size=num_features,
            hidden_size=conv_filters,
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

        inputX = inputX.permute(0, 2, 1)
        inputX = self.cnn(inputX)
        inputX = inputX.permute(0, 2, 1)

        _, (hn, _) = self.lstm(inputX, (h0, c0))
        out = self.relu(self.linear(hn[-1]).flatten())
        return out

    # Autoregression portion of the LSTM
    # For future reference should maybe have to change for when you put in previous SOH into input
    def predict(self, inputX):
        inputX = inputX.permute(0, 2, 1)
        inputX = self.cnn(inputX)
        inputX = inputX.permute(0, 2, 1)
        _, (hn, cn) = self.lstm(inputX, (self.hidden, self.control))
        out = self.sigmoid(self.linear(hn[-1]).flatten())
        self.hidden = hn
        self.control = cn
        return out

# Defined the training loop
if __name__ == "__main__":
    df_train, _, _ = helper.get_dataset("data/battery_log_processed.csv")
    df_train_loader = helper.get_loaders(df_train)

    LSTM_CNN_model = LSTM_CNN(len(list(df_train.columns.difference(["SOH[%]"]))), conv_filters=64, kernel_size=3)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(LSTM_CNN_model.parameters(), lr=0.005)

    f = open("loss2.csv", "w")
    print(" -- Starting classical loss -- ")
    for ix_epoch in range(10):
        print(f"Epoch {ix_epoch}\n---------")

        total_loss = 0
        num_batch = len(df_train_loader)
        for X, y in df_train_loader:
            output = LSTM_CNN_model.forward(X)
            loss = loss_function(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batch
        f.write(avg_loss.__str__() + ",")
        print(f"LSTM Train loss per batch: {avg_loss}")
