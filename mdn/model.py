import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


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
