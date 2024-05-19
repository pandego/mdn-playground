import os
from datetime import datetime

import optuna
from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from dataset import generate_data, get_dataloader, split_data
from model import MDNModel


def objective(trial: optuna.trial.Trial) -> float:
    """
    Define the objective function for the Optuna study.

    Parameters
    ----------
    trial : optuna.trial.Trial
        The trial object.

    Returns
    -------
    float
        The validation loss of the model.

    Notes
    -----
    The objective function generates data, creates a DataLoader, suggests hyperparameters, defines the model, sets up the trainer, trains the model, and returns the validation loss.

    """
    # Define run path for artifacts
    # TODO: Create a path outside src/ and log artifacts into MLflow server
    dataset_name = "example_dataset"
    time_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts_path = f"artifacts/{dataset_name}"
    # run_path = f"{artifacts_path}/{time_now}"  # comment if you use parallelism
    run_reference = 'test_1'
    run_path = f"{artifacts_path}/{run_reference}"  # uncomment if you use parallelism
    os.makedirs(run_path, exist_ok=True)

    # Generate data and create DataLoader
    if dataset_name == "example_dataset":
        (x, y), _ = generate_data()
    else:
        logger.error(NotImplementedError)

    x_train, y_train, x_val, y_val, x_test, y_test = split_data(
        x, y, training_split=0.7, validation_split=0.2
    )
    train_loader = get_dataloader(x_train, y_train, batch_size=32)
    val_loader = get_dataloader(x_val, y_val, batch_size=32, shuffle=False)

    # Suggest hyperparameters
    input_dim, output_dim = 1, 1  # example dimensions
    num_hidden = trial.suggest_int("num_hidden", 20, 100)
    num_mixtures = trial.suggest_int("num_mixtures", 2, 10)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    # Define the model
    model = MDNModel(input_dim, output_dim, num_mixtures, num_hidden, learning_rate)

    # Set up the trainer
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        filename="best-checkpoint",
        dirpath=f"{run_path}/checkpoints",
    )
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=10, mode="min")

    logger_tb = TensorBoardLogger(f"{artifacts_path}/logs", name="optuna_logs")
    logger_csv = CSVLogger(f"{artifacts_path}/logs", name="csv_logs")  # already default

    trainer = Trainer(
        devices=-1,  # if more than 1, consider use 'if trainer.is_global_zero'
        strategy="ddp",  # Use Distributed Data Parallel
        enable_progress_bar=True,
        enable_model_summary=True,
        max_epochs=100,
        # max_epochs=max_epochs,
        # min_epochs=min_epochs,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        callbacks=[checkpoint_callback, early_stopping_callback],
        # logger=True,      # if True, default is CSVLogger, dir='lightning_logs/'
        logger=[logger_tb, logger_csv],
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Get the best score
    return trainer.callback_metrics["val_loss"].item()


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
