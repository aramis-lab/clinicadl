from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch import nn

from clinicadl.utils.metric_module import MetricModule
from clinicadl.utils.network import Network
from clinicadl.utils.network.network_utils import CropMaxUnpool3d, PadMaxPool3d


class AutoEncoder(Network):
    _evaluation_metrics = ["MSE", "MAE"]
    _ensemble_results = False

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

    def predict(self, input):
        _, output = self.forward(input)
        return output

    def forward(self, input):
        indices_list = []
        pad_list = []
        x = input
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

        imgs = input_dict["image"].to(self.device)
        train_output = self.predict(imgs)
        loss = criterion(train_output, imgs)

        return train_output, loss

    @staticmethod
    def _test_columns(mode):
        columns = [
            "participant_id",
            "session_id",
            f"{mode}_id",
            "MSE",
            "MAE",
        ]

        return columns

    @staticmethod
    def _generate_test_row(idx, data, outputs, mode):
        from torch.nn.functional import l1_loss, mse_loss

        device = outputs.device
        image = data["image"][idx].to(device)
        mse_value = mse_loss(outputs[idx], image)
        mae_value = l1_loss(outputs[idx], image)
        row = [
            [
                data["participant_id"][idx],
                data["session_id"][idx],
                data[f"{mode}_id"][idx].item(),
                mse_value,
                mae_value,
            ]
        ]
        return row

    @staticmethod
    def _compute_metrics(results_df):

        return {
            "MSE": results_df.MSE.mean(),
            "MAE": results_df.MAE.mean(),
        }

    @staticmethod
    def _ensemble_fn(
        performance_df,
        validation_df,
        mode,
        selection_threshold=None,
        use_labels=True,
    ):
        raise NotImplementedError(
            "No ensemble method was implemented for AutoEncoder class."
        )


class CNN(Network):
    _evaluation_metrics = ["accuracy", "sensitivity", "specificity", "PPV", "NPV", "BA"]
    _ensemble_results = True

    def __init__(self, convolutions, classifier, use_cpu=False):
        super().__init__(use_cpu=use_cpu)
        self.convolutions = convolutions.to(self.device)
        self.classifier = classifier.to(self.device)

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

    def forward(self, input):
        return self.layers(input)

    def predict(self, input):
        return self.layers(input)

    def compute_outputs_and_loss(self, input_dict, criterion, use_labels=True):

        imgs, labels = input_dict["image"].to(self.device), input_dict["label"].to(
            self.device
        )
        train_output = self.forward(imgs)
        if use_labels:
            loss = criterion(train_output, labels)
        else:
            loss = torch.Tensor([0])

        return train_output, loss

    @staticmethod
    def _test_columns(mode):
        columns = [
            "participant_id",
            "session_id",
            f"{mode}_id",
            "true_label",
            "predicted_label",
            "proba0",
            "proba1",
        ]

        return columns

    @staticmethod
    def _generate_test_row(idx, data, outputs, mode):
        from torch.nn.functional import softmax

        # TODO: process only idx values
        _, predicted = torch.max(outputs.data, 1)
        normalized_output = softmax(outputs, dim=1)
        row = [
            [
                data["participant_id"][idx],
                data["session_id"][idx],
                data[f"{mode}_id"][idx].item(),
                data["label"][idx].item(),
                predicted[idx].item(),
                normalized_output[idx, 0].item(),
                normalized_output[idx, 1].item(),
            ]
        ]
        return row

    @staticmethod
    def _compute_metrics(results_df):

        metrics_module = MetricModule(CNN.evaluation_metrics)
        return metrics_module.apply(
            results_df.true_label.values.astype(int),
            results_df.predicted_label.values.astype(int),
        )

    @staticmethod
    def _ensemble_fn(
        performance_df,
        validation_df,
        mode,
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
            proba0 = np.average(subject_df["proba0"], weights=weight_series)
            proba1 = np.average(subject_df["proba1"], weights=weight_series)
            proba_list = [proba0, proba1]
            prediction = proba_list.index(max(proba_list))
            label = subject_df["true_label"].unique().item()

            row = [[subject, session, 0, label, prediction, proba0, proba1]]
            row_df = pd.DataFrame(row, columns=columns)
            df_final = df_final.append(row_df)

        if use_labels:
            results = CNN._compute_metrics(df_final)
        else:
            results = None

        return df_final, results
