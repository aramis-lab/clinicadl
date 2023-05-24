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
        kl_weight=1,
    ):
        super(BaseVAE, self).__init__(gpu=gpu)

        self.lambda1 = recons_weight
        self.lambda2 = kl_weight
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

    # Forward
    def forward(self, x):
        mu, logVar = self.encode(x)
        z = self.reparameterize(mu, logVar)
        return self.decode(z), mu, logVar

    def compute_outputs_and_loss(self, input_dict, criterion, use_labels=False):
        images = input_dict["image"].to(self.device)
        recon_images, mu, logVar = self.forward(images)

        losses = criterion(images, recon_images, mu, logVar)
        reconstruction_loss, kl_loss = losses[0], losses[1]
        total_loss = self.lambda1 * reconstruction_loss + self.lambda2 * kl_loss
        # In the case there is a regularization term
        if len(losses) > 2:
            regularization = losses[2]
            total_loss = (
                self.lambda1 * reconstruction_loss
                + self.lambda2 * kl_loss
                + regularization
            )

        loss_dict = {
            "loss": total_loss,
            "recon_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

        return recon_images, loss_dict

    # VAE specific
    def encode(self, x):
        h = self.encoder(x)
        mu, logVar = self.mu_layer(h), self.var_layer(h)
        return mu, logVar

    def decode(self, z):
        z = self.decoder(z)
        return z

    def reparameterize(self, mu, logVar):
        std = torch.exp(0.5 * logVar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
