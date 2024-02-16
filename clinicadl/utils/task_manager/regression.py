import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import sampler
from torch.utils.data.distributed import DistributedSampler

from clinicadl.utils.exceptions import ClinicaDLArgumentError
from clinicadl.utils.task_manager.task_manager import TaskManager


class RegressionManager(TaskManager):
    def __init__(
        self,
        mode,
    ):
        super().__init__(mode)

    @property
    def columns(self):
        return [
            "participant_id",
            "session_id",
            f"{self.mode}_id",
            "true_label",
            "predicted_label",
        ]

    @property
    def evaluation_metrics(self):
        return ["R2_score", "MAE", "RMSE"]

    @property
    def save_outputs(self):
        return False

    def generate_test_row(self, idx, data, outputs):
        return [
            [
                data["participant_id"][idx],
                data["session_id"][idx],
                data[f"{self.mode}_id"][idx].item(),
                data["label"][idx].item(),
                outputs[idx].item(),
            ]
        ]

    def compute_metrics(self, results_df, report_ci):
        return self.metrics_module.apply(
            results_df.true_label.values,
            results_df.predicted_label.values,
            report_ci=report_ci,
        )

    @staticmethod
    def generate_label_code(df, label):
        return None

    @staticmethod
    def output_size(input_size, df, label):
        return 1

    @staticmethod
    def generate_sampler(
        dataset, sampler_option="random", n_bins=5, dp_degree=None, rank=None
    ):
        df = dataset.df

        count = np.zeros(n_bins)
        values = df[dataset.label].values.astype(float)
        thresholds = [
            min(values) + i * (max(values) - min(values)) / n_bins
            for i in range(n_bins)
        ]
        for idx in df.index:
            label = df.loc[idx, dataset.label]
            key = max(np.where((label >= np.array(thresholds))[0]))
            count[[key]] += 1
        weight_per_class = 1 / np.array(count)
        weights = []

        for idx, label in enumerate(df[dataset.label].values):
            key = max(np.where((label >= np.array(thresholds)))[0])
            weights += [weight_per_class[key]] * dataset.elem_per_image

        if sampler_option == "random":
            if dp_degree is not None and rank is not None:
                return DistributedSampler(
                    weights, num_replicas=dp_degree, rank=rank, shuffle=True
                )
            else:
                return sampler.RandomSampler(weights)
        elif sampler_option == "weighted":
            if dp_degree is not None and rank is not None:
                length = len(weights) // dp_degree + int(
                    rank < len(weights) % dp_degree
                )
            else:
                length = len(weights)
            return sampler.WeightedRandomSampler(weights, length)
        else:
            raise NotImplementedError(
                f"The option {sampler_option} for sampler on regression task is not implemented"
            )

    def ensemble_prediction(
        self,
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

        n_modes = validation_df[f"{self.mode}_id"].nunique()
        weight_series = np.ones(n_modes)

        # Sort to allow weighted average computation
        performance_df.sort_values(
            ["participant_id", "session_id", f"{self.mode}_id"], inplace=True
        )

        # Soft majority vote
        df_final = pd.DataFrame(columns=self.columns)
        for (subject, session), subject_df in performance_df.groupby(
            ["participant_id", "session_id"]
        ):
            label = subject_df["true_label"].unique().item()
            prediction = np.average(
                subject_df["predicted_label"], weights=weight_series
            )
            row = [[subject, session, 0, label, prediction]]
            row_df = pd.DataFrame(row, columns=self.columns)
            df_final = pd.concat([df_final, row_df])

        if use_labels:
            results = self.compute_metrics(df_final, report_ci=False)
        else:
            results = None

        return df_final, results

    @staticmethod
    def get_criterion(criterion=None):
        compatible_losses = [
            "L1Loss",
            "MSELoss",
            "KLDivLoss",
            "BCEWithLogitsLoss",
            "HuberLoss",
            "SmoothL1Loss",
        ]
        if criterion is None:
            return nn.MSELoss()
        if criterion not in compatible_losses:
            raise ClinicaDLArgumentError(
                f"Regression loss must be chosen in {compatible_losses}."
            )
        return getattr(nn, criterion)()

    @staticmethod
    def get_default_network():
        return "Conv5_FC3"
