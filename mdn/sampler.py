# -*- coding: utf-8 -*-
import numpy as np
import torch

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def get_sample_mode(alpha: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """
    Selects the mode of the mixture components for each data point.

    Parameters
    ----------
    alpha : torch.Tensor
        The mixture component weights for each data point.
        Shape (N, K), where:
            N is the number of data points.
            K is the number of mixture components.
    mu : torch.Tensor
        The means of the mixture components for each data point.
        Shape (N, K, T), where T is the dimensionality of the data.

    Returns
    -------
    torch.Tensor
        The mode of the mixture components for each data point. Shape (N, T).

    """
    _, max_component = torch.max(alpha, 1)
    cond_mode = mu[torch.arange(alpha.shape[0]), max_component]
    return cond_mode


def get_sampled_preds(
    alpha: torch.Tensor, sigma: torch.Tensor, mu: torch.Tensor, samples: int = 10
) -> torch.Tensor:
    """
    Samples predictions from the mixture model for each data point.

    Parameters
    ----------
    alpha : torch.Tensor
        The mixture component weights for each data point.
        Shape (N, K), where:
            N is the number of data points.
            K is the number of mixture components.
    sigma : torch.Tensor
        The standard deviations of the mixture components for each data point.
        Shape (N, K, T), where T is the dimensionality of the data.
    mu : torch.Tensor
        The means of the mixture components for each data point.
        Shape (N, K, T), where T is the dimensionality of the data.
    samples : int, optional
        The number of samples to draw for each data point. Default is 10.

    Returns
    -------
    torch.Tensor
        The sampled predictions for each data point. Shape (N, samples, T).

    """
    N, K, T = mu.shape
    sampled_preds = torch.zeros(N, samples, T)
    for i in range(N):
        for j in range(samples):
            u = np.random.uniform()
            cum_alpha = 0
            for k in range(K):
                cum_alpha += alpha[i, k]
                if u < cum_alpha:
                    sampled_preds[i, j] = torch.normal(mu[i, k], sigma[i, k])
                    break
    return sampled_preds
