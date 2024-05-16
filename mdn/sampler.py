import numpy as np
import torch


def sample_mode(alpha: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
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
    out = torch.zeros_like(mu[:, 0, :])
    for i in range(alpha.shape[0]):
        out[i] = mu[i, max_component[i], :]
    return out


def sample_preds(alpha: torch.Tensor, sigma: torch.Tensor, mu: torch.Tensor,
                 samples: int = 10) -> torch.Tensor:
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
    out = torch.zeros(N, samples, T)
    for i in range(N):
        for j in range(samples):
            u = np.random.uniform()
            prob_sum = 0
            for k in range(K):
                prob_sum += alpha[i, k]
                if u < prob_sum:
                    out[i, j] = torch.normal(mu[i, k], sigma[i, k])
                    break
    return out
