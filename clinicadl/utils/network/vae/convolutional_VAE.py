""" Credits to Benoit Sauty de Chalon @bsauty """

from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utilities.general_settings import Settings

from clinicadl.utils.network.network import Network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CVAE_3D(Network):
    """
    This is the convolutional autoencoder whose main objective is to project the MRI into a smaller space
    with the sole criterion of correctly reconstructing the data. Nothing longitudinal here.
    """

    def __init__(self, use_cpu):
        super(CVAE_3D, self).__init__(use_cpu=use_cpu)
        nn.Module.__init__(self)
        self.beta = 5
        self.gamma = 10
        self.lr = 1e-4  # For epochs between MCMC steps
        self.epoch = 0

        # Encoder
        self.conv1 = nn.Conv3d(1, 32, 3, stride=2, padding=1)  # 32 x 40 x 48 x 40
        self.conv2 = nn.Conv3d(32, 64, 3, stride=2, padding=1)  # 64 x 20 x 24 x 20
        self.conv3 = nn.Conv3d(64, 128, 3, stride=2, padding=1)  # 128 x 10 x 12 x 10
        # self.conv4 = nn.Conv3d(128, 128, 3, stride=1, padding=1)            # 256 x 10 x 12 x 10
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)
        # self.bn4 = nn.BatchNorm3d(128)
        self.fc10 = nn.Linear(153600, Settings().dimension)
        self.fc11 = nn.Linear(153600, Settings().dimension)

        # Decoder
        self.fc2 = nn.Linear(Settings().dimension, 307200)
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
        h5 = F.relu(self.fc2(encoded)).reshape([encoded.size()[0], 256, 10, 12, 10])
        h6 = F.relu(self.bn5(self.upconv1(h5)))
        h7 = F.relu(self.bn6(self.upconv2(h6)))
        # h8 = F.relu(self.bn7(self.upconv3(h7)))
        reconstructed = F.relu(self.upconv4(h7))
        return reconstructed

    def reparametrize(self, mu, logVar):
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2).to(device)
        eps = torch.normal(
            mean=torch.tensor([0 for i in range(std.shape[1])]).float(), std=1
        ).to(device)
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
