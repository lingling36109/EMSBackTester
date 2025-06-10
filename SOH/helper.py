import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# Dataset class for the LSTM classes
class SequenceDataset(Dataset):
    # Constructor for the time series data for our RNN
    def __init__(self, df, target, features, sequence_length):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length

        self.y = torch.tensor(df[self.target].values).float()
        self.X = torch.tensor(df[self.features].values).float()

    # Returns total length of time series
    def __len__(self):
        return self.X.shape[0]

    # Grabs next item in the dataset
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
    dataset = dataset.drop(['Counter', 'Time'], axis=1)
    plt.figure(figsize=(30, 30))
    corr = dataset.corr()
    heatmap = sb.heatmap(corr, vmin=-1, vmax=1, cmap="Blues", annot=True)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 12}, pad=12)
    plt.show()


# Just some function tests
if __name__ == '__main__':
    df_train, df_validation, df_test = get_dataset("battery_log_processed.csv")
    print_heat_map(df_train)
    df_train_loader = get_loaders(df_train)
    df_validation_loader = get_loaders(df_validation)
    df_test_loader = get_loaders(df_test)
    print("Done")
