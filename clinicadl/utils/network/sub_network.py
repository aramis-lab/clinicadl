from collections import OrderedDict
from logging import getLogger

import torch
from torch import nn

from clinicadl.utils.exceptions import ClinicaDLNetworksError
from clinicadl.utils.network.network import Network
from clinicadl.utils.network.network_utils import (
    CropMaxUnpool2d,
    CropMaxUnpool3d,
    PadMaxPool2d,
    PadMaxPool3d,
    ReverseLayerF,
)

logger = getLogger("clinicadl.networks")


class AutoEncoder(Network):
    def __init__(self, encoder, decoder, gpu=False):
        super().__init__(gpu=gpu)
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
            raise ClinicaDLNetworksError(
                f"Cannot transfer weights from {transfer_class} to Autoencoder."
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

        return train_output, {"loss": loss}


class CNN(Network):
    def __init__(self, convolutions, fc, n_classes, gpu=False):
        super().__init__(gpu=gpu)
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
            raise ClinicaDLNetworksError(
                f"Can not transfer weights from {transfer_class} to CNN."
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

        return train_output, {"loss": loss}


class CNN_SSDA(Network):
    def __init__(
        self,
        convolutions,
        fc_class_source,
        fc_class_target,
        fc_domain,
        n_classes,
        gpu=False,
    ):
        super().__init__(gpu=gpu)
        self.convolutions = convolutions.to(self.device)
        self.fc_class_source = fc_class_source.to(self.device)
        self.fc_class_target = fc_class_target.to(self.device)
        self.fc_domain = fc_domain.to(self.device)
        self.n_classes = n_classes

    @property
    def layers(self):
        return nn.Sequential(
            self.convolutions,
            self.fc_class_source,
            self.fc_class_target,
            self.fc_domain,
        )

    def transfer_weights(self, state_dict, transfer_class):
        if issubclass(transfer_class, CNN_SSDA):
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
            raise ClinicaDLNetworksError(
                f"Cannot transfer weights from {transfer_class} to CNN."
            )

    def forward(self, x, alpha):
        x = self.convolutions(x)
        x_class_source = self.fc_class_source(x)
        x_class_target = self.fc_class_target(x)
        x_reverse = ReverseLayerF.apply(x, alpha)
        x_domain = self.fc_domain(x_reverse)
        return x_class_source, x_class_target, x_domain

    def predict(self, x):
        return self.forward(x)

    def compute_outputs_and_loss_test(self, input_dict, criterion, alpha, target):
        images, labels = input_dict["image"].to(self.device), input_dict["label"].to(
            self.device
        )
        train_output_source, train_output_target, _ = self.forward(images, alpha)

        if target:
            out = train_output_target
            loss_bce = criterion(train_output_target, labels)

        else:
            out = train_output_source
            loss_bce = criterion(train_output_source, labels)

        return out, {"loss": loss_bce}

    def compute_outputs_and_loss(
        self, data_source, data_target, data_target_unl, criterion, alpha
    ):
        images, labels = (
            data_source["image"].to(self.device),
            data_source["label"].to(self.device),
        )

        images_target, labels_target = (
            data_target["image"].to(self.device),
            data_target["label"].to(self.device),
        )

        images_target_unl = data_target_unl["image"].to(self.device)

        (
            train_output_class_source,
            _,
            train_output_domain_s,
        ) = self.forward(images, alpha)

        (
            _,
            train_output_class_target,
            train_output_domain_t,
        ) = self.forward(images_target, alpha)

        _, _, train_output_domain_target_unlab = self.forward(images_target_unl, alpha)

        loss_classif_source = criterion(train_output_class_source, labels)
        loss_classif_target = criterion(train_output_class_target, labels_target)

        loss_classif = loss_classif_source + loss_classif_target

        labels_domain_s = (
            torch.zeros(data_source["image"].shape[0]).long().to(self.device)
        )

        labels_domain_tl = (
            torch.ones(data_target["image"].shape[0]).long().to(self.device)
        )

        labels_domain_tu = (
            torch.ones(data_target_unl["image"].shape[0]).long().to(self.device)
        )

        loss_domain_lab = criterion(train_output_domain_s, labels_domain_s)
        loss_domain_lab_t = criterion(train_output_domain_t, labels_domain_tl)
        loss_domain_t_unl = criterion(
            train_output_domain_target_unlab, labels_domain_tu
        )

        loss_domain = loss_domain_lab + loss_domain_lab_t + loss_domain_t_unl

        total_loss = loss_classif + 0.1 * loss_domain

        return (
            train_output_class_source,
            train_output_class_target,
            {"loss": total_loss},
        )

    def lr_scheduler(self, lr, optimizer, p):
        lr = lr / (1 + 10 * p) ** 0.75
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return optimizer
