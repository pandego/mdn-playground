# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import generate_data
from loguru import logger
from model import MDNModel


def sample_mode(pi, mu):
    _, max_component = torch.max(pi, 1)
    out = torch.zeros_like(mu[:, 0, :])
    for i in range(pi.shape[0]):
        out[i] = mu[i, max_component[i], :]
    return out


def sample_preds(pi, sigma, mu, samples=10):
    N, K, T = mu.shape
    out = torch.zeros(N, samples, T)
    for i in range(N):
        for j in range(samples):
            u = np.random.uniform()
            prob_sum = 0
            for k in range(K):
                prob_sum += pi[i, k]
                if u < prob_sum:
                    out[i, j] = torch.normal(mu[i, k], sigma[i, k])
                    break
    return out


if __name__ == "__main__":
    logger.info("Evaluating model...")

    # Generate test data
    (x_train, y_train), x_test = generate_data()

    # If you want to load from CSV, uncomment and modify the following line:
    # x_train, y_train = load_data_from_csv('path_to_csv', 'target_column')

    model = MDNModel.load_from_checkpoint(
        checkpoint_path="checkpoints/best-checkpoint.ckpt"
    )
    model.eval()

    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    pi, sigma, mu = model(x_test_tensor)

    cond_mode = sample_mode(pi, mu)
    preds = sample_preds(pi, sigma, mu, samples=10)

    plt.figure(figsize=(8, 8))
    title = "Conditional Mode"
    plt.plot(x_train, y_train, "go", alpha=0.5, markerfacecolor="none")
    plt.plot(x_test, cond_mode.detach().numpy(), "r.")
    plt.title(title)
    plt.savefig(f"artifacts/eval_{title}.png")
    plt.close()

    plt.figure(figsize=(8, 8))
    title = "Means"
    plt.plot(x_train, y_train, "go", alpha=0.5, markerfacecolor="none")
    plt.plot(x_test, mu.detach().numpy().reshape(-1, 30), "r.")
    plt.title(title)
    plt.savefig(f"artifacts/eval_{title}.png")
    plt.close()

    plt.figure(figsize=(8, 8))
    title = "Sampled Predictions"
    plt.plot(x_train, y_train, "go", alpha=0.5, markerfacecolor="none")
    for i in range(preds.shape[1]):
        plt.plot(x_test, preds[:, i, :].detach().numpy(), "r.", alpha=0.3)
    plt.title(title)
    plt.savefig(f"artifacts/eval_{title}.png")
    plt.close()
