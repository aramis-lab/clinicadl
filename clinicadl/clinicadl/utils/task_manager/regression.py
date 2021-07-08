import numpy as np
import pandas as pd

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

    def ensemble_prediction(
        self,
        performance_df,
        validation_df,
        selection_threshold=None,
        use_labels=True,
        method="soft",
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
        weight_series = pd.DataFrame(np.ones((n_modes, 1)))

        # Sort to allow weighted average computation
        performance_df.sort_values(
            ["participant_id", "session_id", f"{self.mode}_id"], inplace=True
        )
        weight_series.sort_index(inplace=True)

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
