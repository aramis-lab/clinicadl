""" Credits to Benoit Sauty de Chalon @bsauty """

import torch
import torch.nn as nn
import torch.nn.functional as F

from clinicadl.utils.exceptions import ClinicaDLArgumentError
from clinicadl.utils.network.network import Network
from clinicadl.utils.network.vae.vae_utils import multiply_list


class CVAE_3D(Network):
    """
    This is the convolutional autoencoder whose main objective is to project the MRI into a smaller space
    with the sole criterion of correctly reconstructing the data. Nothing longitudinal here.
    """

    def __init__(self, latent_space_size, gpu):
        super(CVAE_3D, self).__init__(gpu=gpu)
        nn.Module.__init__(self)
        self.beta = 5

        # Encoder
        # Input size 1 x 169 x 208 x 179
        self.conv1 = nn.Conv3d(1, 32, 3, stride=2, padding=1)  # 32 x 85 x 104 x 90
        self.conv2 = nn.Conv3d(32, 64, 3, stride=2, padding=1)  # 64 x 43 x 52 x 45
        self.conv3 = nn.Conv3d(64, 128, 3, stride=2, padding=1)  # 128 x 22 x 26 x 23
        # self.conv4 = nn.Conv3d(128, 128, 3, stride=1, padding=1)            # 256 x 10 x 12 x 10
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)
        self.bn4 = nn.BatchNorm3d(128)
        self.fc10 = nn.Linear(1683968, latent_space_size)
        self.fc11 = nn.Linear(1683968, latent_space_size)

        # Decoder
        self.fc2 = nn.Linear(latent_space_size, 3367936)
        self.upconv1 = nn.ConvTranspose3d(
            256, 128, 3, stride=2, padding=1, output_padding=[0, 1, 0]
        )  # 64 x 10 x 12 x 10
        self.upconv2 = nn.ConvTranspose3d(
            128, 64, 3, stride=2, padding=1, output_padding=[0, 1, 1]
        )  # 64 x 20 x 24 x 20
        # self.upconv3 = nn.ConvTranspose3d(64, 32, 3, stride=1, padding=1)                     # 32 x 40 x 48 x 40
        self.upconv4 = nn.ConvTranspose3d(
            64, 1, 3, stride=2, padding=1, output_padding=[0, 1, 0]
        )  # 1 x 80 x 96 x 80
        self.bn5 = nn.BatchNorm3d(128)
        self.bn6 = nn.BatchNorm3d(64)
        self.bn7 = nn.BatchNorm3d(32)

        self.to(self.device)

    def encoder(self, image):
        h1 = F.relu(self.bn1(self.conv1(image)))
        h2 = F.relu(self.bn2(self.conv2(h1)))
        h3 = F.relu(self.bn3(self.conv3(h2)))
        # h4 = F.relu(self.bn4(self.conv4(h3)))
        # h5 = F.relu(self.fc1(h4.flatten(start_dim=1)))
        h5 = h3.flatten(start_dim=1)
        mu = torch.tanh(self.fc10(h5))
        logVar = self.fc11(h5)
        return mu, logVar

    def decoder(self, encoded):
        h5 = F.relu(self.fc2(encoded)).reshape([encoded.size()[0], 256, 22, 26, 23])
        h6 = F.relu(self.bn5(self.upconv1(h5)))
        h7 = F.relu(self.bn6(self.upconv2(h6)))
        # h8 = F.relu(self.bn7(self.upconv3(h7)))
        reconstructed = F.relu(self.upconv4(h7))
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

    def loss(self, mu, logVar, reconstructed, input_):
        kl_divergence = (
            0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp()) / mu.shape[0]
        )
        # recon_error = torch.nn.MSELoss(reduction='mean')(reconstructed, input_)
        recon_error = torch.sum((reconstructed - input_) ** 2) / input_.shape[0]
        return recon_error, kl_divergence

    def predict(self, x):
        self.training = False
        _, _, output = self.forward(x)
        return output

    def compute_outputs_and_loss(self, input_dict, criterion, use_labels=False):
        vae_criterion = self.loss

        self.training = True

        input_ = input_dict["image"].to(self.device)
        mu, logVar, reconstructed = self.forward(input_)
        reconstruction_loss, kl_loss = vae_criterion(mu, logVar, input_, reconstructed)

        loss_dict = {
            "loss": reconstruction_loss + self.beta * kl_loss,
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


class CVAE_3D_half(Network):
    """
    This is the convolutional autoencoder whose main objective is to project the MRI into a smaller space
    with the sole criterion of correctly reconstructing the data. Nothing longitudinal here.
    """

    def __init__(
        self,
        size_reduction_factor,
        latent_space_size,
        gpu,
        recons_weight,
        kl_weight,
        # normalization,
    ):
        super(CVAE_3D_half, self).__init__(gpu=gpu)
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
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)
        self.bn4 = nn.BatchNorm3d(128)

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
        self.bn5 = nn.BatchNorm3d(128)
        self.bn6 = nn.BatchNorm3d(64)
        self.bn7 = nn.BatchNorm3d(32)

        # if normalization == "batch":
        #     self.norm1 = nn.BatchNorm3d(32)
        #     self.norm2 = nn.BatchNorm3d(64)
        #     self.norm3 = nn.BatchNorm3d(128)
        #     # self.norm4 = nn.BatchNorm3d(128)
        #     self.norm5 = nn.BatchNorm3d(128)
        #     self.norm6 = nn.BatchNorm3d(64)
        #     self.norm7 = nn.BatchNorm3d(32)
        # elif normalization == "group":
        #     self.norm1 = nn.GroupNorm(6, 32)
        #     self.norm2 = nn.GroupNorm(6, 64)
        #     self.norm3 = nn.GroupNorm(6, 128)
        #     # self.norm4 = nn.GroupNorm(128)
        #     self.norm5 = nn.GroupNorm(6, 128)
        #     self.norm6 = nn.GroupNorm(6, 64)
        #     self.norm7 = nn.GroupNorm(6, 32)
        # elif normalization == "instance":
        #     self.norm1 = nn.InstanceNorm3d(32)
        #     self.norm2 = nn.InstanceNorm3d(64)
        #     self.norm3 = nn.InstanceNorm3d(128)
        #     # self.norm4 = nn.InstanceNorm3d(128)
        #     self.norm5 = nn.InstanceNorm3d(128)
        #     self.norm6 = nn.InstanceNorm3d(64)
        #     self.norm7 = nn.InstanceNorm3d(32)
        # else:
        #     raise ClinicaDLArgumentError(
        #         f"{normalization} is an unknown normalization method. Please choose between 'batch', 'group' or 'instance'."
        #     )

        self.to(self.device)

    def encoder(self, image):
        h1 = F.leaky_relu(self.bn1(self.conv1(image)), negative_slope=0.2, inplace=True)
        h2 = F.leaky_relu(self.bn2(self.conv2(h1)), negative_slope=0.2, inplace=True)
        h3 = F.leaky_relu(self.bn3(self.conv3(h2)), negative_slope=0.2, inplace=True)
        # h4 = F.relu(self.bn4(self.conv4(h3)))
        # h5 = F.relu(self.fc1(h4.flatten(start_dim=1)))
        h5 = h3.flatten(start_dim=1)
        mu = torch.tanh(self.fc10(h5))
        logVar = self.fc11(h5)
        return mu, logVar

    def decoder(self, encoded):
        h5 = F.relu(self.fc2(encoded)).reshape(
            [
                encoded.size()[0],
                256,
                self.input_size[1] // 2**self.n_conv,
                self.input_size[2] // 2**self.n_conv,
                self.input_size[3] // 2**self.n_conv,
            ]
        )
        h6 = F.relu(self.bn5(self.upconv1(h5)))
        h7 = F.relu(self.bn6(self.upconv2(h6)))
        # h8 = F.relu(self.bn7(self.upconv3(h7)))
        reconstructed = torch.sigmoid(self.upconv4(h7))
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
