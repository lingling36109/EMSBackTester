import os
import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from torch import nn
from abc import ABC
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# Abstract base class for all autoregressive type models
class base(ABC, nn.Module):
    def __init__(self):
        super().__init__()
        self.control = None
        self.hidden = None

    def predict(self, inputX):
        raise NotImplementedError

    def init_hidden(self, batch_size):
        self.hidden = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        self.control = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()


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
    modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    datasetPath = os.path.join(modpath, filePth)
    df = pd.read_csv(datasetPath)
    df = df.drop(['Counter', 'Time'], axis=1)

    train_size = int(0.8 * len(df))
    validation_size = int(0.1 * len(df))

    df_train = df.loc[:train_size].copy()
    df_validation = df.loc[train_size:train_size + validation_size].copy()
    df_test = df.loc[train_size + validation_size:].copy()

    return df_train, df_validation, df_test


# Change the dataframes into data loaders
def get_loaders(df, batch_size=32, sequence_length=64):
    dataset_loader = SequenceDataset(
        df,
        target="SOH[%]",
        features=list(df.columns.difference(["SOH[%]"])),
        sequence_length=sequence_length
    )

    return DataLoader(dataset_loader, batch_size=batch_size, shuffle=True)


# Prints correlation heat map for the given dataset
def print_heat_map(dataset):
    plt.figure(figsize=(30, 30))
    corr = dataset.corr()
    heatmap = sb.heatmap(corr, vmin=-1, vmax=1, cmap="Blues", annot=True)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 12}, pad=12)
    plt.show()


# Training function for all base classes
def trainer(train_data, valid_data, model, csvFile, batch_size, lr=0.001):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_valid_loss = np.inf
    patience = 0

    f = open(csvFile, "w")
    print(" -- Starting classical loss -- ")
    for ix_epoch in range(100):
        print(f"Epoch {ix_epoch}\n---------")

        total_loss = 0
        num_batch = len(train_data)
        for X, y in train_data:
            output = model.forward(X)
            loss = loss_function(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        valid_loss = tester(valid_data, model, batch_size)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("Model improved; saved to best_model.pth")
        else:
            patience += 1
        if patience >= 5:
            break

        avg_loss = total_loss / num_batch
        f.write(avg_loss.__str__() + ",")
        print(f"Train loss per batch: {avg_loss}")


# Testing function for all base classes
def tester(data_loader, model, batch_size, writeOut=False):
    f = open("data/losses/valid.csv", "w")
    loss_function = nn.MSELoss()
    num_batch = len(data_loader)
    total_loss = 0

    model.init_hidden(batch_size=batch_size)

    with torch.no_grad():
        for X, y in data_loader:
            output = model.predict(X)
            loss = loss_function(output, y)
            total_loss += loss.item()

    avg_loss = total_loss / num_batch
    print(f"Test loss: {avg_loss}")

    if writeOut:
        f.write(avg_loss.__str__() + ",")

    return avg_loss


# Just some function tests
if __name__ == '__main__':
    df_train, df_validation, df_test = get_dataset("data/battery_log_processed.csv")
    print_heat_map(df_train)
    df_train_loader = get_loaders(df_train)
    df_validation_loader = get_loaders(df_validation)
    df_test_loader = get_loaders(df_test)
    print("Done")
