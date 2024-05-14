import lightning.pytorch.loggers
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import torch
from loguru import logger
from dataset import generate_data, get_dataloader
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


if __name__ == '__main__':
    (x, y), x_test = generate_data()

    # Shuffle the data
    np.random.seed(42)
    idx = np.random.permutation(len(x))
    x, y = x[idx], y[idx]

    # Split the data into train and val sets
    x_train, y_train = x[:2000], y[:2000]
    x_val, y_val = x[2000:], y[2000:]

    plt.scatter(x_train, y_train, s=10, alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Train Set")
    plt.savefig('train_data.png')
    plt.close()

    plt.scatter(x_val, y_val, s=10, alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Val Set")
    plt.savefig('val_data.png')
    plt.close()

    train_loader = get_dataloader(
        x_train,
        y_train,
        batch_size=32,
        num_workers=15
    )
    val_loader = get_dataloader(
        x_val,
        y_val,
        batch_size=32,
        shuffle=False,
        num_workers=15
    )

    # Define the model
    input_dim, output_dim, num_mixtures = x_train.shape[1], y_train.shape[1], 3

    model = MDNModel(
        input_dim=input_dim,
        output_dim=output_dim,
        num_mixtures=num_mixtures
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        filename='best-checkpoint',
        dirpath='checkpoints/'
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )

    logger_tb = TensorBoardLogger('artifacts/', name='lightning_logs')
    logger_csv = CSVLogger('./', name='csv_logs')  # already default

    trainer = pl.Trainer(
        devices=4,
        max_epochs=3000,
        min_epochs=50,
        log_every_n_steps=100,
        callbacks=[checkpoint_callback, early_stopping_callback],
        # logger=True,  # if True, default is CSVLogger, dir='lightning_logs/'
        logger=[logger_tb, logger_csv]
    )
    trainer.fit(model, train_loader, val_loader)

    # Save the model
    trainer.save_checkpoint("artifacts/mdn_model.ckpt")

    # Load the model for inference
    model = MDNModel.load_from_checkpoint(
        checkpoint_path="checkpoints/best-checkpoint.ckpt",
        input_dim=1,
        output_dim=1,
        num_mixtures=3,
    )

    model.eval()
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    pi, sigma, mu = model(x_test_tensor)

    cond_mode = sample_mode(pi, mu)
    preds = sample_preds(pi, sigma, mu, samples=10)

    plt.figure(figsize=(8, 8))
    title = "Conditional Mode"
    plt.plot(x_train, y_train, 'go', alpha=0.5, markerfacecolor='none')
    plt.plot(x_test, cond_mode.detach().numpy(), 'r.')
    plt.title(title)
    plt.savefig(f'artifacts/{title}.png')
    plt.close()

    plt.figure(figsize=(8, 8))
    title = "Means"
    plt.plot(x_train, y_train, 'go', alpha=0.5, markerfacecolor='none')
    plt.plot(x_test, mu.detach().numpy().reshape(-1, num_mixtures), 'r.')
    plt.title(title)
    plt.savefig(f'artifacts/{title}.png')
    plt.close()

    plt.figure(figsize=(8, 8))
    title = "Sampled Predictions"
    plt.plot(x_train, y_train, 'go', alpha=0.5, markerfacecolor='none')
    for i in range(preds.shape[1]):
        plt.plot(x_test, preds[:, i, :].detach().numpy(), 'r.', alpha=0.3)
    plt.title(title)
    plt.savefig(f'artifacts/{title}.png')
    plt.close()
