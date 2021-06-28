import abc

import numpy as np
import pandas as pd
import torch
from torch import nn

from clinicadl.utils.metric_module import MetricModule


class Network(nn.Module):
    """Abstract Template for all networks used in ClinicaDL"""

    def __init__(self, use_cpu=False):
        super(Network, self).__init__()
        # TODO: check if gpu is available
        self.device = self._select_device(use_cpu)
        self.metrics_module = MetricModule(self.evaluation_metrics)

    @staticmethod
    def _select_device(use_cpu):
        import os

        from numpy import argmax

        if use_cpu:
            return "cpu"
        else:
            # TODO: check on cluster (add try except)
            # Add option gpu_device (user chooses the gpu)
            # How to perform multi-GPU ?
            # Use get device properties de pytorch instead of nvidia-smi
            os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp")
            memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
            free_gpu = argmax(memory_available)
            return f"cuda:{free_gpu}"

    def predict(self, input):
        return self.layers(input)

    @abc.abstractmethod
    def forward(self, input):
        pass

    @abc.abstractmethod
    def compute_outputs_and_loss(self, input_dict, criterion, use_labels=True):
        pass

    @abc.abstractproperty
    def layers(self):
        pass

    @abc.abstractproperty
    def ensemble_prediction(self):
        """If True results on parts of images can be merged to find a result at the image level."""
        pass

    @abc.abstractproperty
    def evaluation_metrics(self):
        pass

    def test(self, dataloader, criterion, mode="image", use_labels=True):
        """
        Computes the predictions and evaluation metrics.

        Args:
            dataloader: (DataLoader) wrapper of a dataset.
            criterion: (loss) function to calculate the loss.
            mode: (str) input used by the network. Chosen from ['image', 'patch', 'roi', 'slice'].
            use_labels (bool): If True the true_label will be written in output DataFrame and metrics dict will be created.
        Returns
            (DataFrame) results of each input.
            (dict) ensemble of metrics + total loss on mode level.
        """
        self.eval()
        dataloader.dataset.eval()

        columns = self._test_columns(mode)

        results_df = pd.DataFrame(columns=columns)
        total_loss = 0
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                outputs, loss = self.compute_outputs_and_loss(
                    data, criterion, use_labels=use_labels
                )
                total_loss += loss.item()

                # Generate detailed DataFrame
                for idx in range(len(data["participant_id"])):
                    row = self._generate_test_row(idx, data, outputs, mode, use_labels)
                    row_df = pd.DataFrame(row, columns=columns)
                    results_df = pd.concat([results_df, row_df])

                del outputs, loss
            results_df.reset_index(inplace=True, drop=True)

        if not use_labels:
            metrics_dict = None
        else:
            metrics_dict = self._compute_metrics(results_df)
            metrics_dict["loss"] = total_loss
        torch.cuda.empty_cache()

        return results_df, metrics_dict

    @abc.abstractmethod
    def _test_columns(self, mode, use_labels):
        pass

    @abc.abstractmethod
    def _generate_test_row(self, idx, data, outputs, mode, use_labels=True):
        pass

    @abc.abstractmethod
    def _compute_metrics(self, results_df):
        pass

    @abc.abstractmethod
    def _soft_voting(
        self,
        performance_df,
        validation_df,
        mode,
        selection_threshold=None,
        use_labels=True,
    ):
        pass


class CNN(Network):
    def __init__(self, convolutions, classifier, use_cpu=False):
        super().__init__(use_cpu=use_cpu)
        self.convolutions = convolutions.to(self.device)
        self.classifier = classifier.to(self.device)

    @property
    def layers(self):
        return nn.Sequential(self.convolutions, self.classifier)

    @property
    def ensemble_prediction(self):
        return True

    @property
    def evaluation_metrics(self):
        return ["accuracy", "sensitivity", "specificity", "PPV", "NPV", "BA"]

    def forward(self, input):
        return self.predict(input)

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

    def _test_columns(self, mode):
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

    def _generate_test_row(self, idx, data, outputs, mode, use_labels=True):
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

    def _compute_metrics(self, results_df):

        return self.metrics_module.apply(
            results_df.true_label.values.astype(int),
            results_df.predicted_label.values.astype(int),
        )

    def _soft_voting(
        self,
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
        columns = self._test_columns(mode="image")
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
            results = self._compute_metrics(df_final)
        else:
            results = None

        return df_final, results
