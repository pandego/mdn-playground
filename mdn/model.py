import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from rich import print as rprint


class MDN(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden, num_mixtures):
        super(MDN, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, num_hidden),
            nn.Tanh()
        )
        self.z_alpha = nn.Linear(num_hidden, num_mixtures)
        self.z_sigma = nn.Linear(num_hidden, num_mixtures * output_dim)
        self.z_mu = nn.Linear(num_hidden, num_mixtures * output_dim)
        self.num_mixtures = num_mixtures
        self.output_dim = output_dim

    def forward(self, x):
        hidden = self.hidden(x)
        alpha = F.softmax(self.z_alpha(hidden), dim=-1)
        sigma = torch.exp(self.z_sigma(hidden)).view(-1, self.num_mixtures,
                                                     self.output_dim)
        mu = self.z_mu(hidden).view(-1, self.num_mixtures, self.output_dim)
        return alpha, sigma, mu


def mdn_loss(alpha, sigma, mu, target, eps=1e-8):
    target = target.unsqueeze(1).expand_as(mu)
    sigma = sigma + eps  # Add epsilon to avoid log(0) or division by zero
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    log_prob = m.log_prob(target)
    log_prob = log_prob.sum(dim=2)
    log_alpha = torch.log(alpha + eps)  # Add epsilon to avoid log(0)
    loss = -torch.logsumexp(log_alpha + log_prob, dim=1)
    return loss.mean()


class MDNModel(pl.LightningModule):
    def __init__(self, input_dim, output_dim, num_mixtures, num_hidden,
                 learning_rate=1e-3):
        super(MDNModel, self).__init__()
        self.model = MDN(input_dim, output_dim, num_hidden, num_mixtures)
        self.learning_rate = learning_rate

        # Save hyperparameters to ensure they are saved in the checkpoint
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        alpha, sigma, mu = self(x)
        loss = mdn_loss(alpha, sigma, mu, y)
        self.log('train_loss', loss)

        # TODO: Implement logging Histograms and Images
        hist_and_img = False  # not fully implemented yet!
        if hist_and_img:
            # Log histogram of alpha values
            self.logger.experiment.add_histogram('alpha', alpha, self.global_step)
            self.logger.experiment.add_histogram('sigma', sigma, self.global_step)
            self.logger.experiment.add_histogram('mu', mu, self.global_step)

            if batch_idx == 0:  # Log images once per epoch
                sample_mode_image = self.sample_mode_image(alpha, mu)
                self.logger.experiment.add_image('sample_mode', sample_mode_image,
                                                 self.global_step, dataformats='HWC')

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        alpha, sigma, mu = self(x)
        loss = mdn_loss(alpha, sigma, mu, y)
        self.log('val_loss', loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    # TODO: Implement sample_mode_image method below
    def sample_mode_image(self, alpha, mu):
        _, max_component = torch.max(alpha, 1)
        out = torch.zeros_like(mu[:, 0, :])

        for i in range(alpha.shape[0]):
            out[i] = mu[i, max_component[i], :]

        # Normalize the output to the range [0, 1]
        out = (out - out.min()) / (out.max() - out.min())

        # Create a grid of images
        out_image = vutils.make_grid(out.unsqueeze(1), normalize=True)
        return out_image

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
                num_hidden=hparams['num_hidden'],
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
                num_hidden=kwargs['num_hidden'],
                num_mixtures=kwargs['num_mixtures'],
                learning_rate=kwargs.get('learning_rate', 1e-3)
            )

        # Load state dict
        model.load_state_dict(checkpoint['state_dict'])

        return model
