"""
USAD: UnSupervised Anomaly Detection (KDD 2020)
Core model: Adversarial Autoencoder Architecture
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, output_dim // 4),
            nn.ReLU(),
            nn.Linear(output_dim // 4, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class USAD(nn.Module):
    """
    USAD model: shared Encoder E, two independent Decoders D1 and D2.
    AE1 = D1(E(x)), AE2 = D2(E(x))
    """

    def __init__(self, window_size, n_features, latent_dim=40):
        super().__init__()
        input_dim = window_size * n_features
        self.E = Encoder(input_dim, latent_dim)
        self.D1 = Decoder(latent_dim, input_dim)
        self.D2 = Decoder(latent_dim, input_dim)
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.n_features = n_features
        self.window_size = window_size

    def forward_ae1(self, x):
        z = self.E(x)
        return self.D1(z)

    def forward_ae2(self, x):
        z = self.E(x)
        return self.D2(z)

    def forward_ae2_of_ae1(self, x):
        """Two-stage: x -> E -> D1 -> E -> D2"""
        z = self.E(x)
        h1 = self.D1(z)
        z2 = self.E(h1.detach())
        return self.D2(z2)

    def anomaly_score(self, x, alpha=0.5, beta=0.5):
        """
        Compute anomaly score for input window x.
        A(x) = alpha * ||x - AE1(x)||^2 + beta * ||x - AE2(AE1(x))||^2
        """
        ae1_out = self.forward_ae1(x)
        ae2_of_ae1_out = self.forward_ae2_of_ae1(x)

        err1 = torch.mean((x - ae1_out) ** 2, dim=1)
        err2 = torch.mean((x - ae2_of_ae1_out) ** 2, dim=1)

        score = alpha * err1 + beta * err2
        return score, err1, err2

    def reshape_window(self, x):
        """Reshape flat window to (batch, window_size, n_features) for inspection."""
        b = x.size(0)
        return x.view(b, self.window_size, self.n_features)
