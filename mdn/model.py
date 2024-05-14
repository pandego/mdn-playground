import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import yaml
from loguru import logger
from rich import print as rprint


class MDN(nn.Module):
    def __init__(self, input_dim, output_dim, num_mixtures):
        super(MDN, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh()
        )
        self.z_pi = nn.Linear(50, num_mixtures)
        self.z_sigma = nn.Linear(50, num_mixtures * output_dim)
        self.z_mu = nn.Linear(50, num_mixtures * output_dim)
        self.num_mixtures = num_mixtures
        self.output_dim = output_dim

    def forward(self, x):
        hidden = self.hidden(x)
        pi = F.softmax(self.z_pi(hidden), dim=-1)
        sigma = torch.exp(self.z_sigma(hidden)).view(-1, self.num_mixtures,
                                                     self.output_dim)
        mu = self.z_mu(hidden).view(-1, self.num_mixtures, self.output_dim)
        return pi, sigma, mu


def mdn_loss(pi, sigma, mu, target):
    target = target.unsqueeze(1).expand_as(mu)
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    log_prob = m.log_prob(target)
    log_prob = log_prob.sum(dim=2)
    log_pi = torch.log(pi + 1e-10)
    loss = -torch.logsumexp(log_pi + log_prob, dim=1)
    return loss.mean()


class MDNModel(pl.LightningModule):
    def __init__(self, input_dim, output_dim, num_mixtures, learning_rate=1e-3):
        super(MDNModel, self).__init__()
        self.model = MDN(input_dim, output_dim, num_mixtures)
        self.learning_rate = learning_rate

        # Save hyperparameters to ensure they are saved in the checkpoint
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pi, sigma, mu = self(x)
        loss = mdn_loss(pi, sigma, mu, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pi, sigma, mu = self(x)
        loss = mdn_loss(pi, sigma, mu, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, **kwargs):
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)

        # Extract hyperparameters from checkpoint
        try:
            hparams = checkpoint['hyper_parameters']

            model = cls(
                input_dim=hparams['input_dim'],
                output_dim=hparams['output_dim'],
                num_mixtures=hparams['num_mixtures'],
                learning_rate=hparams.get('learning_rate', 1e-3),
            )
            rprint('Hyperparameters in checkpoint. Loading from checkpoint.')

        except KeyError:
            rprint('Hyperparameters not found in checkpoint. Loading defaults.')
            # Create model with arguments passed in kwargs
            model = cls(
                input_dim=kwargs['input_dim'],
                output_dim=kwargs['output_dim'],
                num_mixtures=kwargs['num_mixtures'],
                learning_rate=kwargs.get('learning_rate', 1e-3)
            )

        # Load state dict
        model.load_state_dict(checkpoint['state_dict'])

        return model
