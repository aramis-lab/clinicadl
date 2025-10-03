from typing import Any, Optional

from ..utils.config.base import ClinicaDLConfig


class TrainerState(ClinicaDLConfig):
    """
    Represents the state of a :py:class:`~clinicadl.train.Trainer`.

    Attributes
    ----------
    should_stop : bool
        Whether the training should be stopped at the end of the
        current epoch.
    current_train_batch : int
        Index of the current training batch.
    num_train_batches : int
        Total number of training batches.
    current_val_batch : int
        Index of the current validation batch.
    num_val_batches : int
        Total number of validation batches.
    current_epoch : int
        Index of the current epoch.
    num_epochs : int
        Total number of epochs.
    optim_step : int
        The number of optimization steps performed so far.
    split_idx : int
        Index of the split on which the current model is trained.
    """

    should_stop: bool = False
    current_train_batch: int = 0
    num_train_batches: int = 0
    current_val_batch: int = 0
    num_val_batches: int = 0
    current_pred_batch: int = 0
    num_pred_batches: int = 0
    current_epoch: int = 0
    num_epochs: int = 0
    optim_step: int = 0
    split_idx: Optional[int] = None

    def reset(self) -> None:
        """
        To reset the whole trainer state.
        """
        self.should_stop = False
        self.current_train_batch = 0
        self.current_val_batch = 0
        self.current_pred_batch = 0
        self.current_epoch = 0
        self.optim_step = 0
        self.split_idx = None

    def reset_validation(self) -> None:
        """
        To reset the validation state.
        """
        self.current_val_batch = 0

    def reset_prediction(self) -> None:
        """
        To reset the prediction state.
        """
        self.current_pred_batch = 0

    def state_dict(self) -> dict[str, Any]:
        """
        Returns the trainer state as a dict.

        Returns
        --------
        dict[str, Any]
        """
        return self.to_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        To reload a trainer state given a ``state_dict``.

        Parameters
        ----------
        state_dict : dict[str, Any]
            The trainer state returned by :py:meth:`state_dict`.
        """
        self.__dict__.update(state_dict)
