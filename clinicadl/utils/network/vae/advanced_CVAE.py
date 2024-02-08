import torch
import torch.nn as nn
import torch.nn.functional as F

from clinicadl.utils.network.network import Network
from clinicadl.utils.network.vae.vae_utils import multiply_list


class CVAE_3D_final_conv(Network):
    """
    This is the convolutional autoencoder whose main objective is to project the MRI into a smaller space
    with the sole criterion of correctly reconstructing the data. Nothing longitudinal here.
    fc = final layer conv
    """

    def __init__(
        self, size_reduction_factor, latent_space_size, gpu, recons_weight, kl_weight
    ):
        super(CVAE_3D_final_conv, self).__init__(gpu=gpu)
        nn.Module.__init__(self)
        self.alpha = recons_weight
        self.beta = kl_weight
        self.n_conv = 3
        self.latent_space_size = latent_space_size
        if size_reduction_factor == 2:
            self.input_size = [1, 80, 96, 80]
        elif size_reduction_factor == 3:
            self.input_size = [1, 56, 64, 56]
        elif size_reduction_factor == 4:
            self.input_size = [1, 40, 48, 40]
        elif size_reduction_factor == 5:
            self.input_size = [1, 32, 40, 32]
        self.feature_size = int(
            multiply_list(self.input_size[1:], 2**self.n_conv) * 128
        )

        # Encoder
        self.conv1 = nn.Conv3d(1, 32, 3, stride=2, padding=1)  # 32 x 40 x 48 x 40
        self.conv2 = nn.Conv3d(32, 64, 3, stride=2, padding=1)  # 64 x 20 x 24 x 20
        self.conv3 = nn.Conv3d(64, 128, 3, stride=2, padding=1)  # 128 x 10 x 12 x 10
        # self.conv4 = nn.Conv3d(128, 128, 3, stride=1, padding=1)            # 256 x 10 x 12 x 10
        self.in1 = nn.InstanceNorm3d(32)
        self.in2 = nn.InstanceNorm3d(64)
        self.in3 = nn.InstanceNorm3d(128)
        # self.in4 = nn.InstanceNorm3d(128)
        self.fc10 = nn.Linear(self.feature_size, self.latent_space_size)
        self.fc11 = nn.Linear(self.feature_size, self.latent_space_size)

        # Decoder
        self.fc2 = nn.Linear(self.latent_space_size, 2 * self.feature_size)
        self.upconv1 = nn.ConvTranspose3d(
            256, 128, 3, stride=2, padding=1, output_padding=1
        )  # 64 x 10 x 12 x 10
        self.upconv2 = nn.ConvTranspose3d(
            128, 64, 3, stride=2, padding=1, output_padding=1
        )  # 64 x 20 x 24 x 20
        # self.upconv3 = nn.ConvTranspose3d(64, 32, 3, stride=1, padding=1)                     # 32 x 40 x 48 x 40
        self.upconv4 = nn.ConvTranspose3d(
            64, 1, 3, stride=2, padding=1, output_padding=1
        )  # 1 x 80 x 96 x 80
        self.final = nn.Conv3d(1, 1, 3, stride=1, padding=1)
        self.in5 = nn.InstanceNorm3d(128)
        self.in6 = nn.InstanceNorm3d(64)
        self.in7 = nn.InstanceNorm3d(1)

        self.to(self.device)

    def encoder(self, image):
        h1 = F.leaky_relu(self.in1(self.conv1(image)), negative_slope=0.2, inplace=True)
        h2 = F.leaky_relu(self.in2(self.conv2(h1)), negative_slope=0.2, inplace=True)
        h3 = F.leaky_relu(self.in3(self.conv3(h2)), negative_slope=0.2, inplace=True)
        # h4 = F.relu(self.in4(self.conv4(h3)))
        # h5 = F.relu(self.fc1(h4.flatten(start_dim=1)))
        h5 = h3.flatten(start_dim=1)
        mu = torch.tanh(self.fc10(h5))
        logVar = self.fc11(h5)
        return mu, logVar

    def decoder(self, encoded):
        h5 = F.leaky_relu(self.fc2(encoded)).reshape(
            [
                encoded.size()[0],
                256,
                self.input_size[1] // 2**self.n_conv,
                self.input_size[2] // 2**self.n_conv,
                self.input_size[3] // 2**self.n_conv,
            ]
        )
        h6 = F.leaky_relu(self.in5(self.upconv1(h5)))
        h7 = F.leaky_relu(self.in6(self.upconv2(h6)))
        h8 = F.leaky_relu(self.in7(self.upconv4(h7)))
        reconstructed = torch.sigmoid(self.final(h8))
        return reconstructed

    def reparametrize(self, mu, logVar):
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2).to(self.device)
        eps = torch.normal(
            mean=torch.tensor([0 for i in range(std.shape[1])]).float(), std=1
        ).to(self.device)
        if self.beta != 0:  # beta VAE
            return mu + eps * std
        else:  # regular AE
            return mu

    def forward(self, image):
        mu, logVar = self.encoder(image)
        if self.training:
            encoded = self.reparametrize(mu, logVar)
        else:
            encoded = mu
        reconstructed = self.decoder(encoded)
        return mu, logVar, reconstructed

    def predict(self, x):
        mu, _ = self.encoder(x)
        reconstructed = self.decoder(mu)
        return reconstructed

    def compute_outputs_and_loss(self, input_dict, criterion, use_labels=False):
        input_ = input_dict["image"].to(self.device)
        mu, logVar, reconstructed = self.forward(input_)
        losses = criterion(input_, reconstructed, mu, logVar)
        reconstruction_loss, kl_loss = losses[0], losses[1]
        total_loss = reconstruction_loss + self.beta * kl_loss

        if len(losses) > 2:
            regularization = losses[2]
            total_loss = (
                self.alpha * reconstruction_loss + self.beta * kl_loss + regularization
            )

        loss_dict = {
            "loss": total_loss,
            "recon_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }
        return reconstructed, loss_dict

    @staticmethod
    def get_input_size():
        return "1@169x208x179"

    @staticmethod
    def get_dimension():
        return "3D"

    @staticmethod
    def get_task():
        return ["reconstruction"]
