import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (
            torch.tensor(
                self.x[idx],
                dtype=torch.float32
            ),
            torch.tensor(
                self.y[idx],
                dtype=torch.float32
            )
        )


def generate_data(n=2500):
    x_train = np.random.uniform(0, 1, (n, 1)).astype(np.float32)
    noise = np.random.uniform(-0.1, 0.1, (n, 1)).astype(np.float32)
    y_train = x_train + 0.3 * np.sin(2 * np.pi * x_train) + noise

    # Invert the data as per the paper
    x_train_inv = y_train
    y_train_inv = x_train

    x_test_inv = np.linspace(-0.1, 1.1, n).reshape(-1, 1).astype(np.float32)

    return (x_train_inv, y_train_inv), x_test_inv


def load_data_from_csv(file_path, target_column, delimiter=';'):
    data = pd.read_csv(file_path, delimiter=delimiter)
    x = data.drop(columns=[target_column]).values.astype(np.float32)
    y = data[target_column].values.astype(np.float32).reshape(-1, 1)
    return x, y


def get_dataloader(x, y, batch_size=32, shuffle=True, num_workers=0):
    dataset = CustomDataset(x, y)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader
