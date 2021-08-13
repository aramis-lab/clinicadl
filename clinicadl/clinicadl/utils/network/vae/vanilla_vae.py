import torch
from torch import nn

from clinicadl.utils.network.network import Network
from clinicadl.utils.network.vae.vae_utils import VAE_Decoder, VAE_Encoder


class VanillaVAE(Network):
    def __init__(
        self,
        input_shape,
        feature_size,
        latent_size,
        latent_dim,
        n_conv,
        io_layer_channel,
        train=False,
    ):
        super(VanillaVAE, self).__init__()

        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.feature_size = feature_size
        self.latent_size = latent_size
        self.n_conv = n_conv
        self.io_layer_channel = io_layer_channel

        self.training = train

        self.encoder = VAE_Encoder(
            input_shape=self.input_shape,
            feature_size=self.feature_size,
            latent_dim=self.latent_dim,
            n_conv=self.n_conv,
            first_layer_channels=self.io_layer_channel,
        )

        if self.latent_dim == 1:
            # hidden => mu
            self.fc1 = nn.Linear(self.feature_size, self.latent_size)
            # hidden => logvar
            self.fc2 = nn.Linear(self.feature_size, self.latent_size)
        elif self.latent_dim == 2:
            # hidden => mu
            self.fc1 = nn.Conv2d(
                self.feature_size, self.latent_size, 3, stride=1, padding=1, bias=False
            )
            # hidden => logvar
            self.fc2 = nn.Conv2d(
                self.feature_size, self.latent_size, 3, stride=1, padding=1, bias=False
            )
        else:
            raise AttributeError(
                "Bad latent dimension specified. Latent dimension must be 1 or 2"
            )

        self.decoder = VAE_Decoder(
            input_shape=self.input_shape,
            latent_size=self.latent_size,
            feature_size=self.feature_size,
            latent_dim=self.latent_dim,
            n_conv=self.n_conv,
            last_layer_channels=self.io_layer_channel,
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc1(h), self.fc2(h)
        return mu, logvar

    def decode(self, z):
        z = self.decoder(z)
        return z

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def reparameterize_eval(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def predict(self, x):
        output = self.forward(x)
        return output

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def compute_outputs_and_loss(self, input_dict, criterion, use_labels=False):

        images = input_dict["image"].to(self.device)
        recon_images, mu, log_var = self.forward(images)

        loss = criterion(recon_images, images, mu, log_var)

        return recon_images, loss
