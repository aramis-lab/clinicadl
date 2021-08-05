from torch import nn
from torch.utils.data import sampler

from clinicadl.utils.task_manager.task_manager import TaskManager


class ReconstructionManager(TaskManager):
    def __init__(
        self,
        mode,
    ):
        super().__init__(mode)

    @property
    def columns(self):
        columns = ["participant_id", "session_id", f"{self.mode}_id"]
        for metric in self.evaluation_metrics:
            columns.append(metric)
        return columns

    @property
    def evaluation_metrics(self):
        return ["MSE", "MAE"]

    @property
    def save_outputs(self):
        return True

    def generate_test_row(self, idx, data, outputs):
        y = data["image"][idx]
        y_pred = outputs[idx].cpu()
        metrics = self.metrics_module.apply(y, y_pred)
        row = [
            data["participant_id"][idx],
            data["session_id"][idx],
            data[f"{self.mode}_id"][idx].item(),
        ]
        for metric in self.evaluation_metrics:
            row.append(metrics[metric])
        return [row]

    def compute_metrics(self, results_df):
        metrics = dict()
        for metric in self.evaluation_metrics:
            metrics[metric] = results_df[metric].mean()
        return metrics

    @staticmethod
    def output_size(input_size, df, label):
        return input_size

    @staticmethod
    def generate_label_code(df, label):
        return None

    @staticmethod
    def generate_sampler(dataset, sampler_option="random", n_bins=5):
        df = dataset.df

        weights = [1] * len(df) * dataset.elem_per_image

        if sampler_option == "random":
            return sampler.RandomSampler(weights)
        else:
            raise NotImplementedError(
                f"The option {sampler_option} for sampler on reconstruction task is not implemented"
            )

    def ensemble_prediction(
        self,
        performance_df,
        validation_df,
        selection_threshold=None,
        use_labels=True,
        method="soft",
    ):
        """
        Do not perform any ensemble prediction as it is not possible for reconstruction.

        Args:
            performance_df (pd.DataFrame): results that need to be assembled.
            validation_df (pd.DataFrame): results on the validation set used to compute the performance
                of each separate part of the image.
            selection_threshold (float): with soft-voting method, allows to exclude some parts of the image
                if their associated performance is too low.
            use_labels (bool): If True, metrics are computed and the label column values must be different
                from None.
            method (str): method to assemble the results. Current implementation proposes soft or hard-voting.

        Returns:
            None
        """
        return None, None

    @staticmethod
    def get_criterion():
        return nn.MSELoss()

    @staticmethod
    def get_default_network():
        return "AE_Conv5_FC3"
