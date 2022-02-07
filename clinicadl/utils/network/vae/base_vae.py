import torch

from clinicadl.utils.network.network import Network


class BaseVAE(Network):
    def __init__(
        self,
        encoder,
        decoder,
        mu_layer,
        var_layer,
        latent_size,
        gpu=True,
        is_3D=False,
        recons_weight=1,
        KL_weight=1,
    ):
        super(BaseVAE, self).__init__(gpu=gpu)

        self.lambda1 = recons_weight
        self.lambda2 = KL_weight
        self.latent_size = latent_size
        self.is_3D = is_3D

        self.encoder = encoder.to(self.device)
        self.mu_layer = mu_layer.to(self.device)
        self.var_layer = var_layer.to(self.device)
        self.decoder = decoder.to(self.device)

    @property
    def layers(self):
        return torch.nn.Sequential(
            self.encoder, self.mu_layer, self.var_layer, self.decoder
        )

    # Network specific
    def predict(self, x):
        output, _, _ = self.forward(x)
        return output

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def compute_outputs_and_loss(self, input_dict, criterion, use_labels=False):

        images = input_dict["image"].to(self.device)
        recon_images, mu, log_var = self.forward(images)

        recon_loss = criterion(recon_images, images)
        if self.is_3D:
            kl_loss = -0.5 * torch.mean(
                torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1)
            )
        else:
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        loss = self.lambda1 * recon_loss + self.lambda2 * kl_loss / self.latent_size

        loss_dict = {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

        return recon_images, loss_dict

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
