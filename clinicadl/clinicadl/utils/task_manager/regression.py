import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import sampler

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
        return ["MSE", "MAE"]

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

    def compute_metrics(self, results_df):
        return self.metrics_module.apply(
            results_df.true_label.values,
            results_df.predicted_label.values,
        )

    @staticmethod
    def generate_label_code(df, label):
        return None

    @staticmethod
    def output_size(input_size, df, label):
        return 1

    @staticmethod
    def generate_sampler(dataset, sampler_option="random", n_bins=5):
        df = dataset.df

        count = np.zeros(n_bins)
        values = df[dataset.label].values.astype(float)

        thresholds = [
            min(values) + i * (max(values) - min(values)) / n_bins
            for i in range(n_bins)
        ]
        for idx in df.index:
            label = df.loc[idx, dataset.label]
            key = max(np.where((label >= thresholds))[0])
            count[key] += 1

        weight_per_class = 1 / np.array(count)
        weights = []

        for idx, label in enumerate(df[dataset.label].values):
            key = max(np.where((label >= thresholds))[0])
            weights += [weight_per_class[key]] * dataset.elem_per_image

        if sampler_option == "random":
            return sampler.RandomSampler(weights)
        elif sampler_option == "weighted":
            return sampler.WeightedRandomSampler(weights, len(weights))
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
            df_final = df_final.append(row_df)

        if use_labels:
            results = self.compute_metrics(df_final)
        else:
            results = None

        return df_final, results

    @staticmethod
    def get_criterion():
        return nn.MSELoss()

    @staticmethod
    def get_default_network():
        return "Conv5_FC3"
