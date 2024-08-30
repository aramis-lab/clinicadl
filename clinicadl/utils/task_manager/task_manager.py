from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.cuda.amp import autocast
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader, Sampler

from clinicadl.caps_dataset.data import CapsDataset
from clinicadl.metrics.metric_module import MetricModule
from clinicadl.network.network import Network
from clinicadl.utils import cluster
from clinicadl.utils.enum import (
    ClassificationLoss,
    ClassificationMetric,
    ReconstructionLoss,
    ReconstructionMetric,
    RegressionLoss,
    RegressionMetric,
    Task,
)
from clinicadl.utils.exceptions import ClinicaDLArgumentError

# if network_task == Task.CLASSIFICATION:
# elif network_task == Task.REGRESSION:
# elif network_task == Task.RECONSTRUCTION:


def get_default_network(network_task: Task) -> str:  # return Network
    """Returns the default network to use when no architecture is specified."""
    if network_task == Task.CLASSIFICATION:
        return "Conv5_FC3"
    elif network_task == Task.REGRESSION:
        return "Conv5_FC3"
    elif network_task == Task.RECONSTRUCTION:
        return "AE_Conv5_FC3"


def get_criterion(
    network_task: Union[str, Task], criterion: Optional[str] = None
) -> _Loss:
    """
    Gives the optimization criterion.
    Must check that it is compatible with the task.

    Args:
        criterion: name of the loss as written in Pytorch.
    Raises:
        ClinicaDLArgumentError: if the criterion is not compatible with the task.
    """

    network_task = Task(network_task)

    if network_task == Task.CLASSIFICATION:
        compatible_losses = [e.value for e in ClassificationLoss]
        if criterion is None:
            return nn.CrossEntropyLoss()
        if criterion not in compatible_losses:
            raise ClinicaDLArgumentError(
                f"Classification loss must be chosen in {compatible_losses}."
            )
        return getattr(nn, criterion)()

    elif network_task == Task.REGRESSION:
        compatible_losses = [e.value for e in RegressionLoss]
        if criterion is None:
            return nn.MSELoss()
        if criterion not in compatible_losses:
            raise ClinicaDLArgumentError(
                f"Regression loss must be chosen in {compatible_losses}."
            )
        return getattr(nn, criterion)()

    elif network_task == Task.RECONSTRUCTION:
        compatible_losses = [e.value for e in ReconstructionLoss]
        if criterion is None:
            return nn.MSELoss()
        if criterion not in compatible_losses:
            raise ClinicaDLArgumentError(
                f"Reconstruction loss must be chosen in {compatible_losses}."
            )
        if criterion == "VAEGaussianLoss":
            from clinicadl.network.vae.vae_utils import VAEGaussianLoss

            return VAEGaussianLoss
        elif criterion == "VAEBernoulliLoss":
            from clinicadl.network.vae.vae_utils import VAEBernoulliLoss

            return VAEBernoulliLoss
        elif criterion == "VAEContinuousBernoulliLoss":
            from clinicadl.network.vae.vae_utils import VAEContinuousBernoulliLoss

            return VAEContinuousBernoulliLoss
        return getattr(nn, criterion)()


def output_size(
    network_task: Union[str, Task],
    input_size: Sequence[int],
    df: pd.DataFrame,
    label: str,
) -> Union[int, Sequence[int]]:
    """
    Computes the output_size needed to perform the task.

    Args:
        input_size: size of the input.
        df: meta-data of the training set.
        label: name of the column containing the labels.
    Returns:
        output_size
    """
    network_task = Task(network_task)
    if network_task == Task.CLASSIFICATION:
        label_code = generate_label_code(network_task, df, label)
        return len(label_code)

    elif network_task == Task.REGRESSION:
        return 1

    elif network_task == Task.RECONSTRUCTION:
        return input_size


def generate_label_code(
    network_task: Union[str, Task], df: pd.DataFrame, label: str
) -> Optional[Dict[str, int]]:
    """
    Generates a label code that links the output node number to label value.

    Args:
        df: meta-data of the training set.
        label: name of the column containing the labels.
    Returns:
        label_code
    """

    network_task = Task(network_task)
    if network_task == Task.CLASSIFICATION:
        unique_labels = list(set(getattr(df, label)))
        unique_labels.sort()
        return {str(key): value for value, key in enumerate(unique_labels)}

    elif network_task == Task.REGRESSION or network_task == Task.RECONSTRUCTION:
        return None


# TODO: add function to check that the output size of the network corresponds to what is expected to
#  perform the task
class TaskManager:
    def __init__(self, mode: str, n_classes: int = None):
        self.mode = mode
        self.metrics_module = MetricModule(self.evaluation_metrics, n_classes=n_classes)

    @property
    @abstractmethod
    def columns(self):
        """
        List of the columns' names in the TSV file containing the predictions.
        """
        pass

    @property
    @abstractmethod
    def evaluation_metrics(self):
        """
        Evaluation metrics which can be used to evaluate the task.
        """
        pass

    @property
    @abstractmethod
    def save_outputs(self):
        """
        Boolean value indicating if the output values should be saved as tensor for this task.
        """
        pass

    @abstractmethod
    def generate_test_row(
        self, idx: int, data: Dict[str, Any], outputs: Tensor
    ) -> List[List[Any]]:
        """
        Computes an individual row of the prediction TSV file.

        Args:
            idx: index of the individual input and output in the batch.
            data: input batch generated by a DataLoader on a CapsDataset.
            outputs: output batch generated by a forward pass in the model.
        Returns:
            list of items to be contained in a row of the prediction TSV file.
        """
        pass

    @abstractmethod
    def compute_metrics(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute the metrics based on the result of generate_test_row

        Args:
            results_df: results generated based on _results_test_row
        Returns:
            dictionary of metrics
        """
        pass

    @abstractmethod
    def ensemble_prediction(
        self,
        performance_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        selection_threshold: float = None,
        use_labels: bool = True,
        method: str = "soft",
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Compute the results at the image-level by assembling the results on parts of the image.

        Args:
            performance_df: results that need to be assembled.
            validation_df: results on the validation set used to compute the performance
                of each separate part of the image.
            selection_threshold: with soft-voting method, allows to exclude some parts of the image
                if their associated performance is too low.
            use_labels: If True, metrics are computed and the label column values must be different
                from None.
            method: method to assemble the results. Current implementation proposes soft or hard-voting.

        Returns:
            the results and metrics on the image level
        """
        pass

    @staticmethod
    @abstractmethod
    def generate_sampler(
        dataset: CapsDataset,
        sampler_option: str = "random",
        n_bins: int = 5,
        dp_degree: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> Sampler:
        """
        Returns sampler according to the wanted options.

        Args:
            dataset: the dataset to sample from.
            sampler_option: choice of sampler.
            n_bins: number of bins to used for a continuous variable (regression task).
            dp_degree: the degree of data parallelism.
            rank: process id within the data parallelism communicator.
        Returns:
            callable given to the training data loader.
        """
        pass

    def test(
        self,
        model: Network,
        dataloader: DataLoader,
        criterion: _Loss,
        use_labels: bool = True,
        amp: bool = False,
        report_ci=False,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Computes the predictions and evaluation metrics.

        Parameters
        ----------
        model: Network
            The model trained.
        dataloader: DataLoader
            Wrapper of a CapsDataset.
        criterion:  _Loss
            Function to calculate the loss.
        use_labels: bool
            If True the true_label will be written in output DataFrame
            and metrics dict will be created.
        amp: bool
            If True, enables Pytorch's automatic mixed precision.

        Returns
        -------
            the results and metrics on the image level.
        """
        model.eval()
        dataloader.dataset.eval()

        results_df = pd.DataFrame(columns=self.columns)
        total_loss = {}
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                # initialize the loss list to save the loss components
                with autocast(enabled=amp):
                    outputs, loss_dict = model(data, criterion, use_labels=use_labels)

                if i == 0:
                    for loss_component in loss_dict.keys():
                        total_loss[loss_component] = 0
                for loss_component in total_loss.keys():
                    total_loss[loss_component] += loss_dict[loss_component].float()

                # Generate detailed DataFrame
                for idx in range(len(data["participant_id"])):
                    row = self.generate_test_row(idx, data, outputs.float())
                    row_df = pd.DataFrame(row, columns=self.columns)
                    results_df = pd.concat([results_df, row_df])

                del outputs, loss_dict
        dataframes = [None] * dist.get_world_size()
        dist.gather_object(
            results_df, dataframes if dist.get_rank() == 0 else None, dst=0
        )
        if dist.get_rank() == 0:
            results_df = pd.concat(dataframes)
        del dataframes
        results_df.reset_index(inplace=True, drop=True)

        if not use_labels:
            metrics_dict = None
        else:
            metrics_dict = self.compute_metrics(results_df, report_ci=report_ci)
            for loss_component in total_loss.keys():
                dist.reduce(total_loss[loss_component], dst=0)
                loss_value = total_loss[loss_component].item() / cluster.world_size

                if report_ci:
                    metrics_dict["Metric_names"].append(loss_component)
                    metrics_dict["Metric_values"].append(loss_value)
                    metrics_dict["Lower_CI"].append("N/A")
                    metrics_dict["Upper_CI"].append("N/A")
                    metrics_dict["SE"].append("N/A")

                else:
                    metrics_dict[loss_component] = loss_value

        torch.cuda.empty_cache()

        return results_df, metrics_dict

    def test_da(
        self,
        model: Network,
        dataloader: DataLoader,
        criterion: _Loss,
        alpha: float = 0,
        use_labels: bool = True,
        target: bool = True,
        report_ci=False,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Computes the predictions and evaluation metrics.

        Args:
            model: the model trained.
            dataloader: wrapper of a CapsDataset.
            criterion: function to calculate the loss.
            use_labels: If True the true_label will be written in output DataFrame
                and metrics dict will be created.
        Returns:
            the results and metrics on the image level.
        """
        model.eval()
        dataloader.dataset.eval()
        results_df = pd.DataFrame(columns=self.columns)
        total_loss = 0
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                outputs, loss_dict = model.compute_outputs_and_loss_test(
                    data, criterion, alpha, target
                )
                total_loss += loss_dict["loss"].item()

                # Generate detailed DataFrame
                for idx in range(len(data["participant_id"])):
                    row = self.generate_test_row(idx, data, outputs)
                    row_df = pd.DataFrame(row, columns=self.columns)
                    results_df = pd.concat([results_df, row_df])

                del outputs, loss_dict
            results_df.reset_index(inplace=True, drop=True)

        if not use_labels:
            metrics_dict = None
        else:
            metrics_dict = self.compute_metrics(results_df, report_ci=report_ci)
            if report_ci:
                metrics_dict["Metric_names"].append("loss")
                metrics_dict["Metric_values"].append(total_loss)
                metrics_dict["Lower_CI"].append("N/A")
                metrics_dict["Upper_CI"].append("N/A")
                metrics_dict["SE"].append("N/A")

            else:
                metrics_dict["loss"] = total_loss

        torch.cuda.empty_cache()

        return results_df, metrics_dict
