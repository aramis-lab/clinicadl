import torch

from clinicadl.utils.network.network import Network


class BaseVAE(Network):
    def __init__(self):
        super(BaseVAE, self).__init__()

    # Network specific
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

        recon_loss = criterion(recon_images, images, mu, log_var)
        kd_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
        )

        loss = recon_loss + kd_loss

        return recon_images, loss

    # VAE specific
    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = self.mu_layer(h), self.var_layer(h)
        return mu, logvar

    def decode(self, z):
        z = self.decoder(z)
        return z

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
