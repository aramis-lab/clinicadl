from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn.functional import softmax
from torch.utils.data import Sampler, sampler
from torch.utils.data.distributed import DistributedSampler

from clinicadl.caps_dataset.data import CapsDataset
from clinicadl.metrics.metric_module import MetricModule
from clinicadl.trainer.config.train import TrainConfig
from clinicadl.utils.enum import (
    ClassificationLoss,
    ClassificationMetric,
    Mode,
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


# TODO: to put in trainer ? trainer_utils.py ?
def create_training_config(task: Union[str, Task]) -> Type[TrainConfig]:
    """
    A factory function to create a Training Config class suited for the task.
    Parameters
    ----------
    task : Union[str, Task]
        The Deep Learning task (e.g. classification).
    -------
    """
    task = Task(task)
    if task == Task.CLASSIFICATION:
        from clinicadl.trainer.config.classification import (
            ClassificationConfig as Config,
        )
    elif task == Task.REGRESSION:
        from clinicadl.trainer.config.regression import (
            RegressionConfig as Config,
        )
    elif task == Task.RECONSTRUCTION:
        from clinicadl.trainer.config.reconstruction import (
            ReconstructionConfig as Config,
        )
    return Config


# This function is not useful anymore since we introduced config class
# default network will automatically be initialized when running the task
def get_default_network(network_task: Task) -> str:
    """Returns the default network to use when no architecture is specified."""
    task_network_map = {
        Task.CLASSIFICATION: "Conv5_FC3",
        Task.REGRESSION: "Conv5_FC3",
        Task.RECONSTRUCTION: "AE_Conv5_FC3",
    }
    return task_network_map.get(network_task, "Unknown Task")


def get_criterion(
    network_task: Union[str, Task], criterion: Optional[str] = None
) -> nn.Module:
    """
    Gives the optimization criterion.
    Must check that it is compatible with the task.

    Args:
        network_task: Task type as a string or Task enum
        criterion: name of the loss as written in PyTorch.
    Raises:
        ClinicaDLArgumentError: if the criterion is not compatible with the task.
    """

    network_task = Task(network_task)

    def validate_criterion(criterion_name: str, compatible_losses: List[str]):
        if criterion_name not in compatible_losses:
            raise ClinicaDLArgumentError(
                f"Loss must be chosen from {compatible_losses}."
            )
        return getattr(nn, criterion_name)()

    if network_task == Task.CLASSIFICATION:
        compatible_losses = [e.value for e in ClassificationLoss]
        return (
            nn.CrossEntropyLoss()
            if criterion is None
            else validate_criterion(criterion, compatible_losses)
        )

    if network_task == Task.REGRESSION:
        compatible_losses = [e.value for e in RegressionLoss]
        return (
            nn.MSELoss()
            if criterion is None
            else validate_criterion(criterion, compatible_losses)
        )

    if network_task == Task.RECONSTRUCTION:
        compatible_losses = [e.value for e in ReconstructionLoss]
        reconstruction_losses = {
            "VAEGaussianLoss": "VAEGaussianLoss",
            "VAEBernoulliLoss": "VAEBernoulliLoss",
            "VAEContinuousBernoulliLoss": "VAEContinuousBernoulliLoss",
        }

        if criterion in reconstruction_losses:
            from clinicadl.network.vae.vae_utils import (
                VAEBernoulliLoss,
                VAEContinuousBernoulliLoss,
                VAEGaussianLoss,
            )

            return eval(reconstruction_losses[criterion])

        return (
            nn.MSELoss()
            if criterion is None
            else validate_criterion(criterion, compatible_losses)
        )


def output_size(
    network_task: Union[str, Task],
    input_size: Optional[Sequence[int]] = None,
    df: Optional[pd.DataFrame] = None,
    label: Optional[str] = None,
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
        return len(generate_label_code(network_task, df, label))
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
        unique_labels = sorted(set(df[label]))
        return {str(key): value for value, key in enumerate(unique_labels)}

    return None


def evaluation_metrics(network_task: Union[str, Task]):
    """
    Evaluation metrics which can be used to evaluate the task.
    """
    network_task = Task(network_task)
    if network_task == Task.CLASSIFICATION:
        x = [e.value for e in ClassificationMetric]
        x.remove("loss")
        return x
    elif network_task == Task.REGRESSION:
        x = [e.value for e in RegressionMetric]
        x.remove("loss")
        return x
    elif network_task == Task.RECONSTRUCTION:
        x = [e.value for e in ReconstructionMetric]
        x.remove("loss")
        return x
    else:
        raise ValueError("Unknown network task")


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
    mode: Mode,
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
                data[f"{mode.value}_id"][idx].item(),
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
                data[f"{mode.value}_id"][idx].item(),
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
            data[f"{mode.value}_id"][idx].item(),
        ]

        for metric in evaluation_metrics(Task.RECONSTRUCTION):
            row.append(metrics[metric])
        return [row]


def compute_metrics(
    network_task: Union[str, Task],
    results_df: pd.DataFrame,
    metrics_module: Optional[MetricModule] = None,
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
        if metrics_module is not None:
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

            metric_result = str(metric_vals.mean())

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
    mode: str,
    metrics_module: MetricModule,
    n_classes: int,
    network_task: Union[str, Task],
    performance_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    selection_threshold: Optional[float] = None,
    use_labels: bool = True,
    method: Optional[str] = None,
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

    if network_task == Task.CLASSIFICATION:
        if method is None:
            method = "soft"
        return ensemble_prediction_classification(
            network_task,
            mode,
            metrics_module,
            n_classes,
            performance_df,
            validation_df,
            selection_threshold,
            use_labels,
            method,
        )
    elif network_task == Task.REGRESSION:
        if method is None:
            method = "hard"
        return ensemble_prediction_regression(
            network_task,
            mode,
            metrics_module,
            n_classes,
            performance_df,
            validation_df,
            selection_threshold,
            use_labels,
            method,
        )
    elif network_task == Task.RECONSTRUCTION:
        return None, None


def ensemble_prediction_regression(
    network_task,
    mode: str,
    metrics_module: MetricModule,
    n_classes: int,
    performance_df,
    validation_df,
    selection_threshold=None,
    use_labels=True,
    method="hard",
):
    """
    Compute the results at the image-level by assembling the results on parts of the image.

    Args:
        performance_df (pd.DataFrame): results that need to be assembled.
        validation_df (pd.DataFrame): results on the validation set used to compute the performance
            of each separate part of the image.
        selection_threshold (float): with soft-voting method, allows to exclude some parts of the image
            if their associated performance is too low.
        use_labels (bool): If True, metrics are computed and the label column values must be different
            from None.
        method (str): method to assemble the results. Current implementation proposes only hard-voting.

    Returns:
        df_final (pd.DataFrame) the results on the image level
        results (Dict[str, float]) the metrics on the image level
    """

    if method != "hard":
        raise NotImplementedError(
            f"You asked for {method} ensemble method. "
            f"The only method implemented for regression is hard-voting."
        )

    n_modes = validation_df[f"{mode}_id"].nunique()
    weight_series = np.ones(n_modes)

    # Sort to allow weighted average computation
    performance_df.sort_values(
        ["participant_id", "session_id", f"{mode}_id"], inplace=True
    )

    # Soft majority vote
    df_final = pd.DataFrame(columns=columns(network_task, mode, n_classes))
    for (subject, session), subject_df in performance_df.groupby(
        ["participant_id", "session_id"]
    ):
        label = subject_df["true_label"].unique().item()
        prediction = np.average(subject_df["predicted_label"], weights=weight_series)
        row = [[subject, session, 0, label, prediction]]
        row_df = pd.DataFrame(row, columns=columns(network_task, mode, n_classes))
        df_final = pd.concat([df_final, row_df])

    if use_labels:
        results = compute_metrics(
            network_task, df_final, metrics_module, report_ci=False
        )
    else:
        results = None

    return df_final, results


def ensemble_prediction_classification(
    network_task,
    mode: str,
    metrics_module: MetricModule,
    n_classes: int,
    performance_df,
    validation_df,
    selection_threshold=None,
    use_labels=True,
    method="soft",
):
    """
    Computes hard or soft voting based on the probabilities in performance_df. Weights are computed based
    on the balanced accuracies of validation_df.

    ref: S. Raschka. Python Machine Learning., 2015

    Args:
        performance_df (pd.DataFrame): Results that need to be assembled.
        validation_df (pd.DataFrame): Results on the validation set used to compute the performance
            of each separate part of the image.
        selection_threshold (float): with soft-voting method, allows to exclude some parts of the image
            if their associated performance is too low.
        use_labels (bool): If True, metrics are computed and the label column values must be different
            from None.
        method (str): method to assemble the results. Current implementation proposes soft or hard-voting.

    Returns:
        df_final (pd.DataFrame) the results on the image level
        results (Dict[str, float]) the metrics on the image level
    """

    def check_prediction(row):
        if row["true_label"] == row["predicted_label"]:
            return 1
        else:
            return 0

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
        weight_series = sub_level_accuracies / sub_level_accuracies.sum()
    elif method == "hard":
        n_modes = validation_df[f"{mode}_id"].nunique()
        weight_series = pd.DataFrame(np.ones((n_modes, 1)))
    else:
        raise NotImplementedError(
            f"Ensemble method {method} was not implemented. "
            f"Please choose in ['hard', 'soft']."
        )

    # Sort to allow weighted average computation
    performance_df.sort_values(
        ["participant_id", "session_id", f"{mode}_id"], inplace=True
    )
    weight_series.sort_index(inplace=True)

    # Soft majority vote
    df_final = pd.DataFrame(columns=columns(network_task, mode, n_classes))
    for (subject, session), subject_df in performance_df.groupby(
        ["participant_id", "session_id"]
    ):
        label = subject_df["true_label"].unique().item()
        proba_list = [
            np.average(subject_df[f"proba{i}"], weights=weight_series)
            for i in range(n_classes)
        ]
        prediction = proba_list.index(max(proba_list))
        row = [[subject, session, 0, label, prediction] + proba_list]
        row_df = pd.DataFrame(row, columns=columns(network_task, mode, n_classes))
        df_final = pd.concat([df_final, row_df])

    if use_labels:
        results = compute_metrics(
            network_task, df_final, metrics_module, report_ci=False
        )
    else:
        results = None

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


# class TaskConfig(BaseModel):
#     mode: str
#     network_task: Task

#     # pydantic config
#     model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)


# class RegressionConfig(TaskConfig):
#     network_task = Task.REGRESSION


# class ReconstructionConfig(TaskConfig):
#     network_task = Task.RECONSTRUCTION


# class ClassificationConfig(TaskConfig):
#     network_task = Task.CLASSIFICATION

#     n_classe: Optional[int] = None
#     df: Optional[pd.DataFrame] = None
#     label: Optional[str] = None

#     @model_validator(mode="after")
#     def model_validator(self):
#         if n_classes is None:
#             n_classes = output_size(Task.CLASSIFICATION, None, df, label)
#         n_classes = n_classes

#         metrics_module = MetricModule(
#             evaluation_metrics(network_task), n_classes=n_classes
#         )
