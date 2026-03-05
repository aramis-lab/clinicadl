from datetime import datetime
from pathlib import Path


class TrainingSummary:
    """
    To create a summary of the training.

    Parameters
    ----------
    path : Path
        The path to the summary file.
    """

    def __init__(self, file: Path):
        self.file = file

    def create(self) -> None:
        """
        Creates a new summary file.
        """
        summary = "================================================ Training ================================================\n\n"
        summary += "Date: " + datetime.now().strftime("%Y %b %d, %H:%M:%S") + "\n"
        with self.file.open(mode="w") as f:
            f.write(summary)

    def add_data_info(self, n_train_samples: int, n_val_samples: int) -> None:
        """
        To add information relative to the training and validation data.

        Parameters
        ----------
        n_train_samples : int
            The number of training samples used during training.
        n_val_samples : int
            The number of validation samples.
        """
        summary = f"\nTrained with {int(n_train_samples):,} samples\n"
        summary += f"Validated on {int(n_val_samples):,} samples\n"
        self.add_info(summary)

    def add_training_end_info(self, n_epochs: int, interrupted: bool) -> None:
        """
        To add information relative to the training end.

        Parameters
        ----------
        n_epochs : int
            The duration of the training, in number of epochs.
        interrupted : False
            Whether the training was interrupted.
        """
        self.add_info(
            f"\nTraining {'interrupted' if interrupted else 'completed'} after {int(n_epochs):,} epochs\n"
        )

    def add_info(self, info: str) -> None:
        """
        To add information in the summary file.

        Parameters
        ----------
        info : str
            The text to add to the summary file.
        """
        with open(self.file, "a") as f:
            f.write(info)
