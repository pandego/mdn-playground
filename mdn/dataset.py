# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


def generate_data(n: int = 2500) -> tuple:
    """
    Generate synthetic data for training and testing.

    Parameters
    ----------
    n : int, optional
        The number of samples to generate, by default 2500

    Returns
    -------
    tuple
        The training and test data.

    Notes
    -----
    The training data is a tuple of input features and target labels.
    The test data is a 1D array of input features.

    """
    x_train = np.random.uniform(0, 1, (n, 1)).astype(np.float32)
    noise = np.random.uniform(-0.1, 0.1, (n, 1)).astype(np.float32)
    y_train = x_train + 0.3 * np.sin(2 * np.pi * x_train) + noise

    # Invert the data as per the paper
    x_train_inv = y_train
    y_train_inv = x_train

    x_test_inv = np.linspace(-0.1, 1.1, n).reshape(-1, 1).astype(np.float32)

    return (x_train_inv, y_train_inv), x_test_inv


def load_data_from_csv(
    file_path: str, target_column: str, delimiter: str = ";"
) -> tuple:
    """
    Load data from a CSV file and split it into features and target labels.

    Parameters
    ----------
    file_path : str
        The path to the CSV file.
    target_column : str
        The name of the column to use as the target labels.
    delimiter : str, optional
        The delimiter used in the CSV file, by default ";"

    Returns
    -------
    tuple
        The features and target labels from the CSV file.

    Notes
    -----
    The features are all columns in the CSV file except the target column.
    The target labels are the values in the target column.

    """
    data = pd.read_csv(file_path, delimiter=delimiter)
    x = data.drop(columns=[target_column]).values.astype(np.float32)
    y = data[target_column].values.astype(np.float32).reshape(-1, 1)
    return x, y


def split_data(
    x: np.ndarray,
    y: np.ndarray,
    training_split: float = 0.7,
    validation_split: float = 0.2,
) -> tuple:
    """
    Split the input data into training, validation, and test sets.

    Parameters
    ----------
    x : np.ndarray
        The input features.
    y : np.ndarray
        The target labels.
    training_split : float, optional
        The proportion of the dataset to include in the training split, by default 0.7
    validation_split : float, optional
        The proportion of the dataset to include in the validation split, by default 0.2

    Returns
    -------
    tuple
        The training, validation, and test sets for both `x` and `y`.

    Notes
    -----
    The remaining data after the training and validation splits will be used for the test set.

    """
    # Shuffle the data prior to splitting
    idx = np.random.permutation(len(x))
    x, y = x[idx], y[idx]

    # Calculate the indices for the splits
    split1 = int(training_split * len(x))
    split2 = split1 + int(validation_split * len(x))

    # Split the data
    x_train, y_train = x[:split1], y[:split1]
    x_val, y_val = x[split1:split2], y[split1:split2]
    x_test, y_test = x[split2:], y[split2:]

    return x_train, y_train, x_val, y_val, x_test, y_test


def get_dataloader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader from the input data.

    Parameters
    ----------
    x : np.ndarray
        The input features.
    y : np.ndarray
        The target labels.
    batch_size : int, optional
        The number of samples per batch, by default 32
    shuffle : bool, optional
        Whether to shuffle the data after each epoch, by default True
    num_workers : int, optional
        The number of worker processes for data loading, by default 0

    Returns
    -------
    DataLoader
        The DataLoader for the input data.

    Notes
    -----
    The DataLoader will provide batches of data from the input features and labels.
    """
    dataset = CustomDataset(x, y)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return dataloader
