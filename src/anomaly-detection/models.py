"""Donut (WWW18) & USAD (KDD20) anomaly detection models in PyTorch."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# USAD — UnSupervised Anomaly Detection (KDD 2020)
# Architecture: shared encoder + two competing decoders
# Anomaly score: alpha * MSE(x, w1) + beta * MSE(x, w2)
# ============================================================

class USADEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(True),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(True),
            nn.Linear(input_dim // 4, latent_dim),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class USADDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, output_dim // 4),
            nn.ReLU(True),
            nn.Linear(output_dim // 4, output_dim // 2),
            nn.ReLU(True),
            nn.Linear(output_dim // 2, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class USAD(nn.Module):
    def __init__(self, window_size, n_features, latent_dim=10):
        super().__init__()
        input_dim = window_size * n_features
        self.window_size = window_size
        self.n_features = n_features
        self.input_dim = input_dim

        self.encoder = USADEncoder(input_dim, latent_dim)
        self.decoder1 = USADDecoder(latent_dim, input_dim)
        self.decoder2 = USADDecoder(latent_dim, input_dim)

    def forward(self, x):
        """x: (batch, window_size * n_features)"""
        z = self.encoder(x)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        return w1, w2, w3

    def anomaly_score(self, x, alpha=0.5, beta=0.5):
        """Compute per-sample anomaly scores."""
        self.eval()
        with torch.no_grad():
            z = self.encoder(x)
            w1 = self.decoder1(z)
            w2 = self.decoder2(self.encoder(w1))
            score = alpha * ((x - w1) ** 2).mean(dim=1) + beta * ((x - w2) ** 2).mean(dim=1)
        return score


# ============================================================
# Donut — VAE for seasonal KPI AD (WWW 2018)
# Architecture: VAE with Gaussian p(z) and q(z|x)
# Anomaly score: negative reconstruction probability
# ============================================================

class DonutEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class DonutDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_logvar = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Donut(nn.Module):
    def __init__(self, window_size, latent_dim=5, hidden_dim=100):
        super().__init__()
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.x_dim = window_size  # univariate, output = window_size

        self.encoder = DonutEncoder(window_size, hidden_dim, latent_dim)
        self.decoder = DonutDecoder(latent_dim, hidden_dim, window_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """x: (batch, window_size) — univariate KPI windows"""
        mu_z, logvar_z = self.encoder(x)
        z = self.reparameterize(mu_z, logvar_z)
        mu_x, logvar_x = self.decoder(z)
        return mu_x, logvar_x, mu_z, logvar_z, z

    def reconstruction_probability(self, x, n_samples=100, reduction='last'):
        """
        Monte Carlo estimate of E_{q(z|x)}[p(x|z)].
        Higher value = more normal. Lower = more anomalous.
        """
        self.eval()
        with torch.no_grad():
            mu_z, logvar_z = self.encoder(x)
            log_probs = []

            for _ in range(n_samples):
                z = self.reparameterize(mu_z, logvar_z)
                mu_x, logvar_x = self.decoder(z)
                log_probs.append(gaussian_log_prob(x, mu_x, logvar_x))

            log_prob_stack = torch.stack(log_probs, dim=0)
            log_mean_prob = torch.logsumexp(log_prob_stack, dim=0) - math.log(n_samples)
            prob = torch.exp(log_mean_prob)

            if reduction == 'last':
                return prob[:, -1]
            if reduction == 'mean':
                return prob.mean(dim=1)
            if reduction == 'none':
                return prob
            raise ValueError(f"Unknown reduction: {reduction}")

    def anomaly_score(self, x, n_samples=100, reduction='last'):
        """Negative avg reconstruction probability — higher = more anomalous."""
        rp = self.reconstruction_probability(x, n_samples, reduction=reduction)
        return -rp


# ============================================================
# Training functions
# ============================================================

def train_usad(model, train_loader, val_loader, epochs, lr=1e-3, device='cpu'):
    model = model.to(device)
    opt1 = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder1.parameters()), lr=lr
    )
    opt2 = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder2.parameters()), lr=lr
    )

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss1, train_loss2 = 0.0, 0.0
        n_batches = 0

        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device)
            n_batches += 1

            # Phase 1: train AE1 (minimize reconstruction of x and w3)
            w1, w2, w3 = model(batch)
            loss1 = (1 / epoch) * F.mse_loss(batch, w1) + (1 - 1 / epoch) * F.mse_loss(batch, w3)
            opt1.zero_grad()
            loss1.backward(retain_graph=True)
            opt1.step()

            # Phase 2: train AE2 (minimize w2 recon, maximize w3 recon)
            w1, w2, w3 = model(batch)
            loss2 = (1 / epoch) * F.mse_loss(batch, w2) - (1 - 1 / epoch) * F.mse_loss(batch, w3)
            opt2.zero_grad()
            loss2.backward()
            opt2.step()

            train_loss1 += loss1.item()
            train_loss2 += loss2.item()

        # Validation
        model.eval()
        val_loss1, val_loss2 = 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(device)
                w1, w2, w3 = model(batch)
                val_loss1 += F.mse_loss(batch, w1).item()
                val_loss2 += F.mse_loss(batch, w2).item()

        n_val = len(val_loader)
        history.append({
            'epoch': epoch,
            'train_loss1': train_loss1 / n_batches,
            'train_loss2': train_loss2 / n_batches,
            'val_loss1': val_loss1 / n_val,
            'val_loss2': val_loss2 / n_val,
        })
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | "
                  f"train AE1: {history[-1]['train_loss1']:.4f}, "
                  f"train AE2: {history[-1]['train_loss2']:.4f} | "
                  f"val AE1: {history[-1]['val_loss1']:.4f}, "
                  f"val AE2: {history[-1]['val_loss2']:.4f}")

    return history


def gaussian_log_prob(x, mu, logvar):
    """Elementwise log probability of x under N(mu, exp(logvar))."""
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    return -0.5 * (
        math.log(2.0 * math.pi) + logvar + (x - mu).pow(2) / logvar.exp()
    )


def donut_loss_fn(x, mu_x, logvar_x, mu_z, logvar_z, mask=None):
    """
    M-ELBO loss for Donut.
    If mask is provided, missing points are excluded from the
    reconstruction term (alpha in the paper).
    """
    # Per-sample KL divergence: D_KL(q(z|x) || p(z)), p(z)=N(0, I).
    kl = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp(), dim=1)

    # Elementwise log probability of x given z.
    log_px_given_z = gaussian_log_prob(x, mu_x, logvar_x)

    if mask is not None:
        if mask.shape != x.shape:
            raise ValueError(f"mask shape {mask.shape} must match x shape {x.shape}")
        alpha = mask.float()
    else:
        alpha = torch.ones_like(x)

    beta = alpha.mean(dim=1)
    recon = (alpha * log_px_given_z).sum(dim=1)
    elbo = recon - beta * kl

    return -elbo.mean()


def train_donut(model, train_loader, val_loader, epochs, lr=1e-3, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device)
            n_batches += 1

            optimizer.zero_grad()
            mu_x, logvar_x, mu_z, logvar_z, z = model(batch)
            loss = donut_loss_fn(batch, mu_x, logvar_x, mu_z, logvar_z)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(device)
                mu_x, logvar_x, mu_z, logvar_z, z = model(batch)
                val_loss += donut_loss_fn(batch, mu_x, logvar_x, mu_z, logvar_z).item()

        n_val = len(val_loader)
        history.append({
            'epoch': epoch,
            'train_loss': train_loss / n_batches,
            'val_loss': val_loss / n_val,
        })
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train loss: {history[-1]['train_loss']:.4f}, "
                  f"val loss: {history[-1]['val_loss']:.4f}")

    return history
