# -*- coding: utf-8 -*-
import argparse
import csv
import os
from datetime import datetime

import numpy as np
import torch
from dataset import generate_data, get_dataloader, load_data_from_csv, split_data
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from loguru import logger
from model import MDNModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sampler import get_sample_mode, get_sampled_preds
from sklearn.metrics import mean_absolute_error, mean_squared_error
from visualization import (
    plot_conditional_mode,
    plot_histogram,
    plot_means,
    plot_sampled_predictions,
    plot_scatter,
    plot_train_val_data,
)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MDN model")
    parser.add_argument("--csv", type=str, help="Path to CSV file")
    parser.add_argument("--target", type=str, help="Target column in CSV file")
    parser.add_argument(
        "--delimiter", type=str, help="Delimiter in CSV file", default=";"
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="Mode to run the script in: 'train' or 'inference'",
        default="train",
        choices=["train", "inference"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    logger.info("Initializing training...!")

    args = parse_args()

    # Load dataset
    if args.csv:
        dataset_name = args.csv.split("/")[-1].split(".")[0]
    else:
        dataset_name = "example_dataset"

    # Define run path for artifacts
    # TODO: Create a path outside src/ and log artifacts into MLflow server
    time_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts_path = f"artifacts/{dataset_name}"
    # run_path = f"{artifacts_path}/{time_now}"  # comment if you use parallelism
    run_reference = "test_1"
    run_path = f"{artifacts_path}/{run_reference}"  # uncomment if you use parallelism
    os.makedirs(run_path, exist_ok=True)

    if args.csv:
        x, y = load_data_from_csv(args.csv, args.target, args.delimiter)
    else:
        (x, y), _ = generate_data()

    training_split = 0.7
    validation_split = 0.2
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(
        x, y, training_split=training_split, validation_split=validation_split
    )

    if not args.csv:
        plot_train_val_data(x_train, y_train, x_val, y_val, run_path)

    # Define the model
    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]
    num_hidden = 50
    num_mixtures = 3
    batch_size = 64
    num_workers = 4
    max_epochs = 3000
    min_epochs = 300

    if args.mode == "train":
        train_loader = get_dataloader(
            x_train, y_train, batch_size=batch_size, num_workers=num_workers
        )
        val_loader = get_dataloader(
            x_val, y_val, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        model = MDNModel(
            input_dim=input_dim,
            output_dim=output_dim,
            num_hidden=num_hidden,
            num_mixtures=num_mixtures,
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            filename="best-checkpoint",
            dirpath=f"{run_path}/checkpoints",
        )

        early_stopping_callback = EarlyStopping(
            monitor="val_loss", patience=10, mode="min"
        )

        logger_tb = TensorBoardLogger(f"{artifacts_path}/logs", name="lightning_logs")
        logger_csv = CSVLogger(
            f"{artifacts_path}/logs", name="csv_logs"
        )  # already default

        trainer = Trainer(
            devices=-1,  # if more than 1, consider use 'if trainer.is_global_zero'
            strategy="ddp",  # Use Distributed Data Parallel
            enable_progress_bar=True,
            enable_model_summary=True,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            log_every_n_steps=10,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
            callbacks=[checkpoint_callback, early_stopping_callback],
            # logger=True,      # if True, default is CSVLogger, dir='lightning_logs/'
            logger=[logger_tb, logger_csv],
        )

        try:
            trainer.fit(model, train_loader, val_loader)
            logger.success("Training completed! 🎉🎉🎉")
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            # exit(1)  # comment if you want to proceed with the saving of the model, etc

        # TODO: This is not the best way to save hyperparameters, but it works for now
        with open(f"{run_path}/hparams.yaml", "w") as f:
            f.write(f"input_dim: {input_dim}\n")
            f.write(f"output_dim: {output_dim}\n")
            f.write(f"num_mixtures: {num_mixtures}\n")
            f.write(f"learning_rate: {model.learning_rate}\n")
            f.write(f"num_hidden: {num_hidden}\n")
            f.write(f"batch_size: 32\n")
            f.write(f"num_workers: 15\n")
            f.write(f"max_epochs: {max_epochs}\n")
            f.write(f"training_split: {training_split}\n")
            f.write(f"validation_split: {validation_split}\n")
            f.write(f"testing_split: {1 - (training_split + validation_split)}\n")

        # Save the model
        trainer.save_checkpoint(f"{run_path}/mdn_model.ckpt")
        logger.success(f"Model saved at {run_path}/mdn_model.ckpt")

    # Load the model for inference
    model = MDNModel.load_from_checkpoint(
        checkpoint_path=f"{run_path}/checkpoints/best-checkpoint.ckpt",
        input_dim=input_dim,
        output_dim=output_dim,
        num_hidden=num_hidden,
        num_mixtures=num_mixtures,
    )

    logger.info(f"Model Summary: \n{model.eval()}")
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    alpha, sigma, mu = model(x_test_tensor)

    cond_mode = get_sample_mode(alpha, mu)

    preds = get_sampled_preds(alpha, sigma, mu, samples=10)

    if not args.csv:
        plot_conditional_mode(x_train, y_train, x_test, cond_mode, run_path)
        plot_means(x_train, y_train, x_test, mu, num_mixtures, run_path)
        plot_sampled_predictions(x_train, y_train, x_test, preds, run_path)

    if args.csv:
        # Flatten the predictions for histogram comparison
        y_pred = preds.mean(axis=1).detach().numpy().flatten()
        y_test_flat = y_test.flatten()

        plot_histogram(y_pred, y_test_flat, run_path, args.target)
        plot_scatter(y_pred, y_test_flat, run_path, args.target)

        # Calculate and log R^2 score
        r2 = np.corrcoef(y_test_flat, y_pred)[0, 1] ** 2
        mae = mean_absolute_error(y_test_flat, y_pred)
        mse = mean_squared_error(y_test_flat, y_pred)
        logger.info(f"R^2 Score: {r2:.4f}")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"MSE: {mse:.4f}")

        # Save R^2 score, MAE, and MSE on a csv with all hyperparameters as columns
        metrics_file = f"{artifacts_path}/metrics_scores.csv"
        file_exists = os.path.isfile(metrics_file)

        # TODO: This is not the best way to save metrics, but it works for now
        with open(metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                # Write header if file does not exist
                writer.writerow(
                    [
                        "run_path",
                        "dataset_name",
                        "r2",
                        "mae",
                        "mse",
                        "num_hidden",
                        "num_mixtures",
                        "learning_rate",
                        "batch_size",
                        "num_workers",
                        "min_epochs",
                        "training_split",
                        "validation_split",
                    ]
                )
            # Write metrics
            writer.writerow(
                [
                    run_path,
                    dataset_name,
                    r2,
                    mae,
                    mse,
                    num_hidden,
                    num_mixtures,
                    model.learning_rate,
                    batch_size,
                    num_workers,
                    min_epochs,
                    training_split,
                    validation_split,
                ]
            )

    logger.info("All done! Have a nice day! 😊")
