from collections import OrderedDict

import torch
from torch import nn

from clinicadl.utils.network.network import Network
from clinicadl.utils.network.network_utils import (
    CropMaxUnpool2d,
    CropMaxUnpool3d,
    PadMaxPool2d,
    PadMaxPool3d,
)


class AutoEncoder(Network):
    def __init__(self, encoder, decoder, use_cpu=False):
        super().__init__(use_cpu=use_cpu)
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)

    @property
    def layers(self):
        return nn.Sequential(self.encoder, self.decoder)

    def transfer_weights(self, state_dict, transfer_class):
        if issubclass(transfer_class, AutoEncoder):
            self.load_state_dict(state_dict)
        elif issubclass(transfer_class, CNN):
            encoder_dict = OrderedDict(
                [
                    (k.replace("convolutions.", ""), v)
                    for k, v in state_dict.items()
                    if "convolutions" in k
                ]
            )
            self.encoder.load_state_dict(encoder_dict)
        else:
            raise ValueError(
                f"Cannot transfer weights from {transfer_class} " f"to Autoencoder."
            )

    def predict(self, x):
        _, output = self.forward(x)
        return output

    def forward(self, x):
        indices_list = []
        pad_list = []
        for layer in self.encoder:
            if (
                (isinstance(layer, PadMaxPool3d) or isinstance(layer, PadMaxPool2d))
                and layer.return_indices
                and layer.return_pad
            ):
                x, indices, pad = layer(x)
                indices_list.append(indices)
                pad_list.append(pad)
            elif (
                isinstance(layer, nn.MaxPool3d) or isinstance(layer, nn.MaxPool2d)
            ) and layer.return_indices:
                x, indices = layer(x)
                indices_list.append(indices)
            else:
                x = layer(x)

        code = x.clone()

        for layer in self.decoder:
            if isinstance(layer, CropMaxUnpool3d) or isinstance(layer, CropMaxUnpool2d):
                x = layer(x, indices_list.pop(), pad_list.pop())
            elif isinstance(layer, nn.MaxUnpool3d) or isinstance(layer, nn.MaxUnpool2d):
                x = layer(x, indices_list.pop())
            else:
                x = layer(x)

        return code, x

    def compute_outputs_and_loss(self, input_dict, criterion, use_labels=True):

        images = input_dict["image"].to(self.device)
        train_output = self.predict(images)
        loss = criterion(train_output, images)

        return train_output, loss


class CNN(Network):
    def __init__(self, convolutions, fc, n_classes, use_cpu=False):
        super().__init__(use_cpu=use_cpu)
        self.convolutions = convolutions.to(self.device)
        self.fc = fc.to(self.device)
        self.n_classes = n_classes

    @property
    def layers(self):
        return nn.Sequential(self.convolutions, self.fc)

    def transfer_weights(self, state_dict, transfer_class):
        if issubclass(transfer_class, CNN):
            self.load_state_dict(state_dict)
        elif issubclass(transfer_class, AutoEncoder):
            convolutions_dict = OrderedDict(
                [
                    (k.replace("encoder.", ""), v)
                    for k, v in state_dict.items()
                    if "encoder" in k
                ]
            )
            self.convolutions.load_state_dict(convolutions_dict)
        else:
            raise ValueError(
                f"Cannot transfer weights from {transfer_class} " f"to CNN."
            )

    def forward(self, x):
        x = self.convolutions(x)
        return self.fc(x)

    def predict(self, x):
        return self.forward(x)

    def compute_outputs_and_loss(self, input_dict, criterion, use_labels=True):

        images, labels = input_dict["image"].to(self.device), input_dict["label"].to(
            self.device
        )
        train_output = self.forward(images)
        if use_labels:
            loss = criterion(train_output, labels)
        else:
            loss = torch.Tensor([0])

        return train_output, loss
