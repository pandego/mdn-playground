# Mixture Density Netwoks - A Playground
## Overview
**Mixture Density Networks (MDNs)** are a class of neural network models introduced by Christopher M. Bishop in his 1994 paper ["Mixture Density Networks"](https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf).
MDNs are designed to overcome the limitations of conventional neural networks when dealing with problems that involve predicting continuous variables, especially in cases where the relationship between input and output is multi-valued.

Here are the key concepts and points from the paper:

### Motivation
#### 1. Limitations of Conventional Neural Networks:

- Conventional neural networks trained with sum-of-squares or cross-entropy error functions approximate the conditional averages of the target data given the input data.
- For classification problems, this approximation works well as it represents the posterior probabilities of class membership.
- However, for problems involving continuous variables, especially in inverse problems where the mapping can be multi-valued, these conditional averages are insufficient. The average of several correct target values is not necessarily a correct value itself.

#### 2. Need for Modeling Conditional Probability Distributions:

- To make accurate predictions for new input vectors, it is crucial to model the complete conditional probability distribution of the target data conditioned on the input vector.
- MDNs address this need by combining a conventional neural network with a mixture density model, allowing them to represent arbitrary conditional probability distributions.

### Mixture Density Network (MDN) Structure
#### 1. Combining Neural Networks and Mixture Models:

- An MDN uses a neural network to determine the parameters of a mixture density model.
- The neural network takes the input vector and outputs parameters for the mixture model, which then represents the conditional probability density of the target variables.

#### 2. Mixture Model:

- The mixture model consists of multiple components. The conditional probability density of the target data $p(t|x)$ is represented as a linear combination of these componentes (or kernel functions) in the following form:

  $$p(t|x) = \sum_{i=1}^{m} \alpha_i(x)\phi_i(t|x)$$

- Bishop has restricted it to kernel functions of Gaussian form, although they are not limited to:

  $$\phi_i(t|x) = \frac{1}{(2\pi)^{\text{c/2}}\sigma_i(x)^c} exp \left\lbrace - \frac{\lVert t-\mu_i(x) \rVert^2}{2\sigma_i(x)^2} \right\rbrace$$

- Each component will have as parameters: mixing coefficients $\alpha_i(x)$, means $\mu_i(x)$, and variances $\sigma_i^2(x)$, which are functions of the input vector $x$.

#### 3. Neural Network Outputs:

- The neural network outputs the parameters for the mixture model.
- Mixing coefficients are determined using a softmax function to ensure they sum to one.
- Variances are represented as exponentials of network outputs to ensure they are positive.
- Means are directly given by the network outputs.

### Training MDNs
#### 1. Error Function:

- The error function for MDNs is derived from the negative logarithm of the likelihood of the data, considering the mixture density model.
- The error function ensures that the network learns the parameters that maximize the likelihood of the data given the model.

#### 2. Optimization:
- Standard optimization techniques like backpropagation can be used to train MDNs.
- Derivatives of the error function with respect to network weights are computed and used to adjust the weights to minimize the error.

### Advantages of MDNs
#### 1. General Framework:

- MDNs provide a general framework for modeling conditional density functions, which can represent complex multi-modal distributions.
- They include the conventional least-squares approach as a special case but are more powerful for complex mappings.

#### 2. Flexibility:

- MDNs can model any conditional density function to arbitrary accuracy by appropriately choosing the mixture model parameters and neural network architecture.

### Implementation Considerations
#### 1. Software Implementation:

- Implementing MDNs in software involves modifying the error function used in conventional neural networks and interpreting the network outputs differently.
- The implementation is straightforward, especially in an object-oriented language like Python.

#### 2. Model Order Selection:

- Selecting the appropriate number of kernel functions in the mixture model is crucial and can be part of the model complexity optimization process.
- Overestimating the number of kernels typically does not degrade performance significantly as redundant kernels can be effectively ignored by the network.

### Takeaways
- Mixture Density Networks provide a powerful and flexible approach to modeling conditional probability distributions in neural networks.
- They are particularly useful for problems with multi-valued mappings and offer a more comprehensive representation of the data compared to traditional methods.

By combining the strengths of neural networks and mixture models, MDNs enable more accurate and reliable predictions for complex real-world applications, as demonstrated in the examples of inverse problems and robot kinematics.




---

## Installation
### 1. Clone the repository and navigate into the directory
```shell
git clone https://github.com/pandego/mdn-playground.git
cd mdn-playground
```

### 2. Setup and activate the conda environment
```shell
condda env create -f environment.yml
conda activate mdn
```

### 4. Install Packages with Poetry
```shell
poetry install
```

### 5. Run a Sample Experiment
```shell
poetry run example
```

---

### TODOs
1. **Experiment Tracking:** Integrate tools like MLflow or Weights & Biases for better experiment tracking and hyperparameter tuning.
2. **Hyperparameter Tuning:** Use libraries like Optuna for automated hyperparameter optimization.
3. **Additional Metrics:** Implement additional evaluation metrics such as Mean Absolute Error (MAE) or Root Mean Squared Error (RMSE).
4. **Model Interpretability:** Explore techniques for interpreting MDNs and understanding the learned conditional probability distributions.
5. **Improved Error Handling:** Implement more robust error handling and logging mechanisms to improve code reliability and maintainability.

---
## Misceallaneous notes
### Sometimes you might see the following warning message:
- GPU with Tensor Cores:
    ```shell
    You are using a CUDA device ('NVIDIA RTX A5000') that has Tensor Cores.
    To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance.
    For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    ```
- Distributed computing:
  ```shell
  The 'train_dataloader' does not have many workers which may be a bottleneck.
  Consider increasing the value of the `num_workers` argument` to `num_workers=15` in the `DataLoader` to improve performance.
  ```
  ```shell
  It is recommended to use `self.log('val_loss', ..., sync_dist=True)` when logging on epoch level in distributed setting to accumulate the metric across devices.
  ```
- Loggers:
  ```shell
   Starting from v1.9.0, `tensorboardX` has been removed as a dependency of the `pytorch_lightning` package, due to potential conflicts with other packages in the ML ecosystem.
   For this reason, `logger=True` will use `CSVLogger` as the default logger, unless the `tensorboard` or `tensorboardX` packages are found.
   Please `pip install lightning[extra]` or one of them to enable TensorBoard support by default
  ```
  ```shell
  The number of training batches (16) is smaller than the logging interval Trainer(log_every_n_steps=50).
  Set a lower value for log_every_n_steps if you want to see logs for the training epoch.
  ```
