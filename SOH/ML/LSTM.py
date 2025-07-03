import helper
import torch
import pandas as pd
from torch import nn

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


# LSTM class
class LSTM(helper.base):
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
        if inputX.shape[0] != self.hidden.shape[1]:
            idx = self.hidden.shape[1] - inputX.shape[0]
            self.hidden = self.hidden[:, idx:]
            self.control = self.control[:, idx:]

        _, (hn, cn) = self.lstm(inputX, (self.hidden, self.control))
        out = self.relu(self.linear(hn[-1]).flatten())
        self.hidden = hn.detach()
        self.control = cn.detach()
        return out

# Defined the training loop
if __name__ == "__main__":
    torch.manual_seed(1)
    df_train, df_valid, df_test = helper.get_dataset("data/training/battery_log_processed.csv")
    df_train_loader = helper.get_loaders(df_train, shuffle=False)
    df_valid_loader = helper.get_loaders(df_valid, shuffle=False)
    df_test_loader = helper.get_loaders(df_test, shuffle=False)

    LSTM_model = LSTM(len(list(df_train[["Avg. Cell V[V]", "Charge State", "Counter", "Avg. Module T[oC]", "Rack Current[A]"]])), hidden_units=32, num_layers=2)
    helper.trainer(df_train_loader, LSTM_model, batch_size=128)

    # df = pd.read_csv("/Users/andrewjosephkim/Desktop/EMSBackTester/SOH/data/battery_log_processed.csv")
    # df_loader = helper.get_loaders(df, shuffle=False)
    #
    # LSTM_model = LSTM(len(list(df[["Avg. Cell V[V]", "Charge State", "Counter", "Avg. Module T[oC]", "Rack Current[A]"]])))
    # LSTM_model.load_state_dict(torch.load("best_model.pth"))
    # helper.tester(df_loader, LSTM_model, 32)
