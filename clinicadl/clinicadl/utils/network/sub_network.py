from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch import nn

from clinicadl.utils.metric_module import MetricModule
from clinicadl.utils.network.network import Network
from clinicadl.utils.network.network_utils import CropMaxUnpool3d, PadMaxPool3d


class AutoEncoder(Network):
    _possible_tasks = ["reconstruction"]

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
                    (k.replace("convolutions", "encoder"), v)
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
                isinstance(layer, PadMaxPool3d)
                and layer.return_indices
                and layer.return_pad
            ):
                x, indices, pad = layer(x)
                indices_list.append(indices)
                pad_list.append(pad)
            elif isinstance(layer, nn.MaxPool3d) and layer.return_indices:
                x, indices = layer(x)
                indices_list.append(indices)
            else:
                x = layer(x)

        code = x.clone()

        for layer in self.decoder:
            if isinstance(layer, CropMaxUnpool3d):
                x = layer(x, indices_list.pop(), pad_list.pop())
            elif isinstance(layer, nn.MaxUnpool3d):
                x = layer(x, indices_list.pop())
            else:
                x = layer(x)

        return code, x

    def compute_outputs_and_loss(self, input_dict, criterion, use_labels=True):

        images = input_dict["image"].to(self.device)
        train_output = self.predict(images)
        loss = criterion(train_output, images)

        return train_output, loss

    @staticmethod
    def _ensemble_fn(
        performance_df,
        validation_df,
        mode,
        evaluation_metrics,
        selection_threshold=None,
        use_labels=True,
    ):
        raise NotImplementedError(
            "No ensemble method was implemented for AutoEncoder class."
        )


class CNN(Network):
    _possible_tasks = ["classification", "regression"]

    def __init__(self, convolutions, classifier, n_classes, use_cpu=False):
        super().__init__(use_cpu=use_cpu)
        self.convolutions = convolutions.to(self.device)
        self.classifier = classifier.to(self.device)
        self.n_classes = n_classes
        assert self.classifier[-1].out_features == n_classes

    @property
    def layers(self):
        return nn.Sequential(self.convolutions, self.classifier)

    def transfer_weights(self, state_dict, transfer_class):
        if issubclass(transfer_class, CNN):
            self.load_state_dict(state_dict)
        elif issubclass(transfer_class, AutoEncoder):
            convolutions_dict = OrderedDict(
                [
                    (k.replace("encoder", "convolutions"), v)
                    for k, v in state_dict.items()
                    if "encoder" in k
                ]
            )
            self.encoder.load_state_dict(convolutions_dict)
        else:
            raise ValueError(
                f"Cannot transfer weights from {transfer_class} " f"to CNN."
            )

    def forward(self, x):
        x = self.convolutions(x)
        return self.classifier(x)

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

    @staticmethod
    def _ensemble_fn(
        performance_df,
        validation_df,
        mode,
        evaluation_metrics,
        selection_threshold=None,
        use_labels=True,
    ):
        """
        Computes soft voting based on the probabilities in performance_df. Weights are computed based on the accuracies
        of validation_df.

        ref: S. Raschka. Python Machine Learning., 2015

        Args:
            performance_df: (DataFrame) results on patch level of the set on which the combination is made.
            validation_df: (DataFrame) results on patch level of the set used to compute the weights.
            mode: (str) input used by the network. Chosen from ['patch', 'roi', 'slice'].
            selection_threshold: (float) if given, all patches for which the classification accuracy is below the
                threshold is removed.
            use_labels: (bool) If True the labels are added to the final TSV file.

        Returns:
            df_final (DataFrame) the results on the image level
            results (dict) the metrics on the image level
        """

        def check_prediction(row):
            if row["true_label"] == row["predicted_label"]:
                return 1
            else:
                return 0

        # Compute the sub-level accuracies on the validation set:
        validation_df["accurate_prediction"] = validation_df.apply(
            lambda x: check_prediction(x), axis=1
        )
        sub_level_accuracies = validation_df.groupby(f"{mode}_id")[
            "accurate_prediction"
        ].sum()
        if selection_threshold is not None:
            sub_level_accuracies[sub_level_accuracies < selection_threshold] = 0
        weight_series = sub_level_accuracies / sub_level_accuracies.sum()
        print(weight_series)

        # Sort to allow weighted average computation
        performance_df.sort_values(
            ["participant_id", "session_id", f"{mode}_id"], inplace=True
        )
        weight_series.sort_index(inplace=True)

        # Soft majority vote
        columns = CNN._test_columns(mode="image")
        df_final = pd.DataFrame(columns=columns)
        for (subject, session), subject_df in performance_df.groupby(
            ["participant_id", "session_id"]
        ):
            print(subject_df)
            proba0 = np.average(subject_df["proba0"], weights=weight_series)
            proba1 = np.average(subject_df["proba1"], weights=weight_series)
            proba_list = [proba0, proba1]
            prediction = proba_list.index(max(proba_list))
            label = subject_df["true_label"].unique().item()

            row = [[subject, session, 0, label, prediction, proba0, proba1]]
            row_df = pd.DataFrame(row, columns=columns)
            df_final = df_final.append(row_df)

        if use_labels:
            results = CNN._compute_metrics(df_final, evaluation_metrics)
        else:
            results = None

        return df_final, results
