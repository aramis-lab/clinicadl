from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from pydantic import (
    BaseModel,
    ConfigDict,
    computed_field,
    model_validator,
)
from torch import Tensor, nn
from torch.cuda.amp import autocast
from torch.nn.functional import softmax
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader, Sampler, sampler
from torch.utils.data.distributed import DistributedSampler

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


# This function is not useful anymore since we introduced config class
# default network will automatically be initialized when running the task
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
) -> nn.Module:
    """
    Gives the optimization criterion.
    Must check that it is compatible with the task.

    Args:
        criterion: name of the loss as written in PyTorch.
    Raises:
        ClinicaDLArgumentError: if the criterion is not compatible with the task.
    """

    def validate_criterion(criterion, compatible_losses):
        if criterion not in compatible_losses:
            raise ClinicaDLArgumentError(
                f"Loss must be chosen from {compatible_losses}."
            )
        return getattr(nn, criterion)()

    def handle_reconstruction_loss(criterion, compatible_losses):
        if criterion == "VAEGaussianLoss":
            from clinicadl.network.vae.vae_utils import VAEGaussianLoss

            return VAEGaussianLoss
        elif criterion == "VAEBernoulliLoss":
            from clinicadl.network.vae.vae_utils import VAEBernoulliLoss

            return VAEBernoulliLoss
        elif criterion == "VAEContinuousBernoulliLoss":
            from clinicadl.network.vae.vae_utils import VAEContinuousBernoulliLoss

            return VAEContinuousBernoulliLoss
        else:
            return validate_criterion(criterion, compatible_losses)

    network_task = Task(network_task)

    if network_task == Task.CLASSIFICATION:
        compatible_losses = [e.value for e in ClassificationLoss]
        return (
            nn.CrossEntropyLoss()
            if criterion is None
            else validate_criterion(criterion, compatible_losses)
        )

    elif network_task == Task.REGRESSION:
        compatible_losses = [e.value for e in RegressionLoss]
        return (
            nn.MSELoss()
            if criterion is None
            else validate_criterion(criterion, compatible_losses)
        )

    elif network_task == Task.RECONSTRUCTION:
        compatible_losses = [e.value for e in ReconstructionLoss]
        return (
            nn.MSELoss()
            if criterion is None
            else handle_reconstruction_loss(criterion, compatible_losses)
        )

    raise ClinicaDLArgumentError("Unknown task type.")


def output_size(
    network_task: Union[str, Task],
    input_size: Optional[Sequence[int]],
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


def evaluation_metrics(network_task: Union[str, Task]):
    """
    Evaluation metrics which can be used to evaluate the task.
    """
    network_task = Task(network_task)
    if network_task == Task.CLASSIFICATION:
        return [e.value for e in ClassificationMetric]
    elif network_task == Task.REGRESSION:
        return [e.value for e in RegressionMetric]
    elif network_task == Task.RECONSTRUCTION:
        return [e.value for e in ReconstructionMetric]


def test(
    network_task,
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

    results_df = pd.DataFrame(columns=columns(network_task, mode, n_classes))
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
                row = generate_test_row(
                    network_task,
                    mode,
                    metrics_module,
                    n_classes,
                    idx,
                    data,
                    outputs.float(),
                )
                row_df = pd.DataFrame(row, columns=self.columns)
                results_df = pd.concat([results_df, row_df])

            del outputs, loss_dict
    dataframes = [None] * dist.get_world_size()
    dist.gather_object(results_df, dataframes if dist.get_rank() == 0 else None, dst=0)
    if dist.get_rank() == 0:
        results_df = pd.concat(dataframes)
    del dataframes
    results_df.reset_index(inplace=True, drop=True)

    if not use_labels:
        metrics_dict = None
    else:
        metrics_dict = compute_metrics(
            network_task, metrics_module, results_df, report_ci=report_ci
        )
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
    network_task: Union[str, Task],
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
    results_df = pd.DataFrame(columns=columns(network_task, mode, n_classes))
    total_loss = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            outputs, loss_dict = model.compute_outputs_and_loss_test(
                data, criterion, alpha, target
            )
            total_loss += loss_dict["loss"].item()

            # Generate detailed DataFrame
            for idx in range(len(data["participant_id"])):
                row = generate_test_row(
                    network_task, mode, metrics_module, n_classes, idx, data, outputs
                )
                row_df = pd.DataFrame(
                    row, columns=columns(network_task, mode, n_classes)
                )
                results_df = pd.concat([results_df, row_df])

            del outputs, loss_dict
        results_df.reset_index(inplace=True, drop=True)

    if not use_labels:
        metrics_dict = None
    else:
        metrics_dict = compute_metrics(
            network_task, metrics_module, results_df, report_ci=report_ci
        )
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


def columns(network_task: Union[str, Task], mode: str, n_classes: Optional[int] = None):
    """
    List of the columns' names in the TSV file containing the predictions.
    """
    network_task = Task(network_task)
    if network_task == Task.CLASSIFICATION:
        return [
            "participant_id",
            "session_id",
            f"{mode}_id",
            "true_label",
            "predicted_label",
        ] + [f"proba{i}" for i in range(n_classes)]
    elif network_task == Task.REGRESSION:
        return [
            "participant_id",
            "session_id",
            f"{mode}_id",
            "true_label",
            "predicted_label",
        ]
    elif network_task == Task.RECONSTRUCTION:
        columns = ["participant_id", "session_id", f"{mode}_id"]
        for metric in evaluation_metrics(network_task):
            columns.append(metric)
        return columns


def save_outputs(network_task: Union[str, Task]):
    """
    Boolean value indicating if the output values should be saved as tensor for this task.
    """

    network_task = Task(network_task)
    if network_task == Task.CLASSIFICATION or network_task == Task.REGRESSION:
        return False
    elif network_task == Task.RECONSTRUCTION:
        return True


def generate_test_row(
    network_task: Union[str, Task],
    mode: str,
    metrics_module,
    n_classes: int,
    idx: int,
    data: Dict[str, Any],
    outputs: Tensor,
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
    network_task = Task(network_task)
    if network_task == Task.CLASSIFICATION:
        prediction = torch.argmax(outputs[idx].data).item()
        normalized_output = softmax(outputs[idx], dim=0)
        return [
            [
                data["participant_id"][idx],
                data["session_id"][idx],
                data[f"{mode}_id"][idx].item(),
                data["label"][idx].item(),
                prediction,
            ]
            + [normalized_output[i].item() for i in range(n_classes)]
        ]

    elif network_task == Task.REGRESSION:
        return [
            [
                data["participant_id"][idx],
                data["session_id"][idx],
                data[f"{mode}_id"][idx].item(),
                data["label"][idx].item(),
                outputs[idx].item(),
            ]
        ]
    elif network_task == Task.RECONSTRUCTION:
        y = data["image"][idx]
        y_pred = outputs[idx].cpu()
        metrics = metrics_module.apply(y, y_pred, report_ci=False)
        row = [
            data["participant_id"][idx],
            data["session_id"][idx],
            data[f"{mode}_id"][idx].item(),
        ]

        for metric in evaluation_metrics(Task.RECONSTRUCTION):
            row.append(metrics[metric])
        return [row]


def compute_metrics(
    network_task: Union[str, Task],
    metrics_module,
    results_df: pd.DataFrame,
    report_ci: bool = False,
) -> Dict[str, float]:
    """
    Compute the metrics based on the result of generate_test_row

    Args:
        results_df: results generated based on _results_test_row
    Returns:
        dictionary of metrics
    """

    network_task = Task(network_task)
    if network_task == Task.CLASSIFICATION or network_task == Task.REGRESSION:
        return metrics_module.apply(
            results_df.true_label.values,
            results_df.predicted_label.values,
            report_ci=report_ci,
        )
    elif network_task == Task.RECONSTRUCTION:
        if not report_ci:
            return {
                metric: results_df[metric].mean()
                for metric in evaluation_metrics(Task.RECONSTRUCTION)
            }

        from numpy import mean as np_mean
        from scipy.stats import bootstrap

        metrics = dict()
        metric_names = ["Metrics"]
        metric_values = ["Values"]
        lower_ci_values = ["Lower bound CI"]
        upper_ci_values = ["Upper bound CI"]
        se_values = ["SE"]

        for metric in evaluation_metrics(Task.RECONSTRUCTION):
            metric_vals = results_df[metric]

            metric_result = metric_vals.mean()

            metric_vals = (metric_vals,)
            # Compute confidence intervals only if there are at least two samples in the data.
            if len(results_df) >= 2:
                res = bootstrap(
                    metric_vals,
                    np_mean,
                    n_resamples=3000,
                    confidence_level=0.95,
                    method="percentile",
                )
                lower_ci, upper_ci = res.confidence_interval
                standard_error = res.standard_error
            else:
                lower_ci, upper_ci, standard_error = "N/A"

            metric_names.append(metric)
            metric_values.append(metric_result)
            lower_ci_values.append(lower_ci)
            upper_ci_values.append(upper_ci)
            se_values.append(standard_error)

        metrics["Metric_names"] = metric_names
        metrics["Metric_values"] = metric_values
        metrics["Lower_CI"] = lower_ci_values
        metrics["Upper_CI"] = upper_ci_values
        metrics["SE"] = se_values

        return metrics


# TODO: add function to check that the output size of the network corresponds to what is expected to
#  perform the task


def ensemble_prediction(
    network_task: Union[str, Task],
    performance_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    selection_threshold: Optional[float] = None,
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

    def check_prediction(row):
        return int(row["true_label"] == row["predicted_label"])

    def calculate_weights_classification(method: str) -> pd.Series:
        if method == "soft":
            # Compute the sub-level accuracies on the validation set:
            validation_df["accurate_prediction"] = validation_df.apply(
                lambda x: check_prediction(x), axis=1
            )
            sub_level_accuracies = validation_df.groupby(f"{mode}_id")[
                "accurate_prediction"
            ].mean()

            if selection_threshold is not None:
                sub_level_accuracies[sub_level_accuracies < selection_threshold] = 0
            return sub_level_accuracies / sub_level_accuracies.sum()

        elif method == "hard":
            n_modes = validation_df[f"{mode}_id"].nunique()
            return pd.Series(np.ones((n_modes, 1)))

        else:
            raise NotImplementedError(
                f"Ensemble method {method} was not implemented. "
                f"Please choose in ['hard', 'soft']."
            )

    def calculate_weights_regression(method: str) -> np.ndarray:
        if method != "hard":
            raise NotImplementedError(
                f"The only method implemented for regression is hard-voting."
            )
        n_modes = validation_df[f"{mode}_id"].nunique()
        return np.ones(n_modes)

    def create_final_dataframe(
        subject_df: pd.DataFrame,
        weight_series: pd.Series,
        label: int,
        is_classification: bool,
    ) -> pd.DataFrame:
        if is_classification:
            proba_list = [
                np.average(subject_df[f"proba{i}"], weights=weight_series)
                for i in range(n_classes)
            ]
            prediction = proba_list.index(max(proba_list))
            row = [[subject, session, 0, label, prediction] + proba_list]
        else:
            prediction = np.average(
                subject_df["predicted_label"], weights=weight_series
            )
            row = [[subject, session, 0, label, prediction]]
        return pd.DataFrame(row, columns=columns(network_task, mode, n_classes))

    network_task = Task(network_task)
    df_final = pd.DataFrame(columns=columns(network_task, mode, n_classes))

    if network_task == Task.CLASSIFICATION:
        weight_series = calculate_weights_classification(method)
    elif network_task == Task.REGRESSION:
        weight_series = calculate_weights_regression(method)
    elif network_task == Task.RECONSTRUCTION:
        return None, None

    performance_df.sort_values(
        ["participant_id", "session_id", f"{mode}_id"], inplace=True
    )
    weight_series.sort_index(inplace=True)

    for (subject, session), subject_df in performance_df.groupby(
        ["participant_id", "session_id"]
    ):
        label = subject_df["true_label"].unique().item()
        df_final = pd.concat(
            [
                df_final,
                create_final_dataframe(
                    subject_df,
                    weight_series,
                    label,
                    network_task == Task.CLASSIFICATION,
                ),
            ]
        )

    results = (
        compute_metrics(network_task, metrics_module, df_final, report_ci=False)
        if use_labels
        else None
    )
    return df_final, results


def generate_sampler(
    network_task: Union[str, Task],
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
        n_bins: number of bins to use for a continuous variable (regression task).
        dp_degree: the degree of data parallelism.
        rank: process id within the data parallelism communicator.
    Returns:
        callable given to the training data loader.
    """

    def calculate_weights_classification(df):
        labels = df[dataset.config.data.label].unique()
        codes = {dataset.config.data.label_code[label] for label in labels}
        count = np.zeros(len(codes))

        for idx in df.index:
            label = df.loc[idx, dataset.config.data.label]
            key = dataset.label_fn(label)
            count[key] += 1

        weight_per_class = 1 / np.array(count)
        weights = [
            weight_per_class[dataset.label_fn(label)] * dataset.elem_per_image
            for label in df[dataset.config.data.label].values
        ]
        return weights

    def calculate_weights_regression(df):
        count = np.zeros(n_bins)
        values = df[dataset.config.data.label].values.astype(float)
        thresholds = np.linspace(min(values), max(values), n_bins, endpoint=False)

        for idx in df.index:
            label = df.loc[idx, dataset.config.data.label]
            key = max(np.where(label >= thresholds)[0])
            count[key] += 1

        weight_per_class = 1 / count
        weights = [
            weight_per_class[max(np.where(label >= thresholds)[0])]
            * dataset.elem_per_image
            for label in df[dataset.config.data.label].values
        ]
        return weights

    def get_sampler(weights):
        if sampler_option == "random":
            if dp_degree is not None and rank is not None:
                return DistributedSampler(
                    weights, num_replicas=dp_degree, rank=rank, shuffle=True
                )
            else:
                return sampler.RandomSampler(weights)
        elif sampler_option == "weighted":
            length = (
                len(weights) // dp_degree + int(rank < len(weights) % dp_degree)
                if dp_degree and rank is not None
                else len(weights)
            )
            return sampler.WeightedRandomSampler(weights, length)
        else:
            raise NotImplementedError(
                f"The option {sampler_option} for sampler is not implemented"
            )

    network_task = Task(network_task)
    df = dataset.df

    if network_task == Task.CLASSIFICATION:
        weights = calculate_weights_classification(df)
    elif network_task == Task.REGRESSION:
        weights = calculate_weights_regression(df)
    elif network_task == Task.RECONSTRUCTION:
        weights = [1] * len(df) * dataset.elem_per_image
    else:
        raise ValueError(f"Unknown network task: {network_task}")

    return get_sampler(weights)


class TaskConfig(BaseModel):
    mode: str
    network_task: Task

    # pydantic config
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)


class RegressionConfig(TaskConfig):
    network_task = Task.REGRESSION


class ReconstructionConfig(TaskConfig):
    network_task = Task.RECONSTRUCTION


class ClassificationConfig(TaskConfig):
    network_task = Task.CLASSIFICATION

    n_classe: Optional[int] = None
    df: Optional[pd.DataFrame] = None
    label: Optional[str] = None

    @model_validator(mode="after")
    def model_validator(self):
        if self.n_classes is None:
            n_classes = output_size(Task.CLASSIFICATION, None, self.df, self.label)
        self.n_classes = n_classes

        self.metrics_module = MetricModule(
            evaluation_metrics(self.network_task), n_classes=self.n_classes
        )
