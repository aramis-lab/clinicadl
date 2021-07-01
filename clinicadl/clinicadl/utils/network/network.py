import abc

import pandas as pd
import torch
from torch import nn

from clinicadl.utils.descriptors import classproperty


class Network(nn.Module):
    """Abstract Template for all networks used in ClinicaDL"""

    def __init__(self, use_cpu=False):
        super(Network, self).__init__()
        # TODO: check if gpu is available
        self.device = self._select_device(use_cpu)

    @classproperty
    def evaluation_metrics(cls):
        return cls._evaluation_metrics

    @classproperty
    def ensemble_results(cls):
        return cls._ensemble_results

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

    @abc.abstractmethod
    def predict(self, input):
        pass

    @abc.abstractmethod
    def forward(self, input):
        pass

    @abc.abstractmethod
    def compute_outputs_and_loss(self, input_dict, criterion, use_labels=True):
        pass

    @property
    @abc.abstractmethod
    def layers(self):
        pass

    @abc.abstractmethod
    def transfer_weights(self, state_dict, transfer_class):
        pass

    def test(self, dataloader, criterion, mode="image", use_labels=True):
        """
        Computes the predictions and evaluation metrics.

        Args:
            dataloader: (DataLoader) wrapper of a dataset.
            criterion: (loss) function to calculate the loss.
            mode: (str) input used by the network. Chosen from ['image', 'patch', 'roi', 'slice'].
            use_labels (bool): If True the true_label will be written in output DataFrame
                and metrics dict will be created.
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
                    row = self._generate_test_row(idx, data, outputs, mode)
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

    @staticmethod
    @abc.abstractmethod
    def _test_columns(mode):
        pass

    @staticmethod
    @abc.abstractmethod
    def _generate_test_row(idx, data, outputs, mode):
        pass

    @staticmethod
    @abc.abstractmethod
    def _compute_metrics(results_df):
        pass

    @staticmethod
    @abc.abstractmethod
    def _ensemble_fn(
        performance_df,
        validation_df,
        mode,
        selection_threshold=None,
        use_labels=True,
    ):
        pass
