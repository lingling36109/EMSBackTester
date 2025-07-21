import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from torch import nn
from abc import ABC
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import root_mean_squared_error


# Abstract base class for all autoregressive type models
class base(ABC, nn.Module):
    def __init__(self):
        super().__init__()
        self.control = None
        self.hidden = None

    def predict(self, inputX):
        raise NotImplementedError

    def init_hidden(self, batch_size):
        self.hidden = torch.zeros(self.num_layers, batch_size, self.hidden_units)
        self.control = torch.zeros(self.num_layers, batch_size, self.hidden_units)


# Dataset class for the LSTM classes
class SequenceDataset(Dataset):
    def __init__(self, df, target, features, sequence_length):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length

        self.y = torch.tensor(df[self.target].values).float()
        self.X = torch.tensor(df[self.features].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if idx >= self.sequence_length - 1:
            start_idx = idx - (self.sequence_length - 1)
            x = self.X[start_idx:(idx + 1)]
            return x, self.y[idx]
        else:
            padding = self.X[0].repeat(self.sequence_length - 1 - idx, 1)
            x = self.X[0:(idx + 1)]
            x = torch.cat([padding, x], dim=0)
            return x, self.y[idx]


# Get the training, validation, and test dataframes (80, 10, 10 split) from original
def get_dataset(filePth):
    df = pd.read_csv("/SOH/data/training/processed/battery_log_processed.csv")
    df = df.drop(['Time'], axis=1)

    train_size = int(0.8 * len(df))
    validation_size = int(0.1 * len(df))

    df_train = df.loc[:train_size].copy()
    df_validation = df.loc[train_size:train_size + validation_size].copy()
    df_test = df.loc[train_size + validation_size:].copy()

    return df_train, df_validation, df_test


# Change the dataframes into data loaders
def get_loaders(df, batch_size=128, sequence_length=1024, shuffle=True):
    dataset_loader = SequenceDataset(
        df,
        target="SOC[%]",
        features=list(df[["Avg. Cell V[V]", "Charge State", "Counter", "Avg. Module T[oC]", "Rack Current[A]"]]),
        sequence_length=sequence_length
    )

    return DataLoader(dataset_loader, batch_size=batch_size, shuffle=shuffle)


# Prints correlation heat map for the given dataset
def print_heat_map(dataset):
    plt.figure(figsize=(30, 30))
    corr = dataset.corr()
    heatmap = sb.heatmap(corr, vmin=-1, vmax=1, cmap="Blues", annot=True)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 12}, pad=12)
    plt.show()
    figure = heatmap.get_figure()
    figure.savefig('/Users/andrewjosephkim/Desktop/EMSBackTester/SOH/saved/heatmap.png', dpi="figure")

# Training function for all base classes
def trainer(train_data, model, batch_size, lr=0.001):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.init_hidden(batch_size=batch_size)

    for ix_epoch in range(40):
        print(f"Epoch Number: {ix_epoch}")
        total_loss = 0
        num_batch = len(train_data)
        for X, y in train_data:
            output = model.forward(X)
            loss = loss_function(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batch
        print(f"Train loss per batch: {avg_loss}")
    torch.save(model.state_dict(), "model.pth")

def tester(data_loader, model, batch_size):
    loss_function = nn.MSELoss()
    num_batch = len(data_loader)
    total_loss = 0

    all_preds = []
    all_targets = []

    model.init_hidden(batch_size=batch_size)

    with torch.no_grad():
        for X, y in data_loader:
            output = model.forward(X)
            loss = loss_function(output, y)
            total_loss += loss.item()

            all_preds.append(output.cpu())
            all_targets.append(y.cpu())

    avg_loss = total_loss / num_batch
    print(f"Test loss: {avg_loss}")

    all_preds = torch.cat(all_preds).squeeze().numpy()
    all_targets = torch.cat(all_targets).squeeze().numpy()

    plt.figure(figsize=(10, 5))
    plt.plot(all_targets, label="Actual", color="black", linewidth=2)
    plt.plot(all_preds, label="Predicted", color="red", linestyle="--")
    plt.title("Model Predictions vs Actual Values")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return avg_loss



def combine_csvs_from_directory(directory_path):
    combined_df = pd.DataFrame()

    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            print(f"Reading {file_path}...")
            df = pd.read_csv(file_path)
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    return combined_df


def save_histograms(df, output_dir="histograms"):
    os.makedirs(output_dir, exist_ok=True)

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            plt.figure()
            df[column].dropna().hist(bins=30, edgecolor='black')
            plt.title(f"Histogram of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.tight_layout()

            filename = f"{column}_histogram.png"
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()


if __name__ == "__main__":
    df = pd.read_csv("/Users/andrewjosephkim/Desktop/EMSBackTester/SOH/dual_ukf_predictions.csv")
    df2 = pd.read_csv("/SOH/data/training/processed/battery_log_processed.csv")

    print(root_mean_squared_error(df['SOC'], df2['Fuck3']))
    # plt.figure(figsize=(20, 10))
    # plt.plot(df['SOC'], label="SOC")
    # plt.plot(df2['Fuck3'], label="SOC Real")
    # plt.legend()
    # plt.savefig('Fuuuuuuck2.png', dpi=450)
    # plt.show()