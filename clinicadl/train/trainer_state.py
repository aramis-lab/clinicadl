from enum import Enum
from typing import Any, Optional

from clinicadl.data.dataloader import DataLoader
from clinicadl.split import Split
from clinicadl.utils.config import ClinicaDLConfig


class TrainerStage(str, Enum):
    """Possible stages of the trainer."""

    TRAIN = "training"
    VAL = "validation"
    TEST = "test"
    PREDICT = "prediction"


class TrainerState(ClinicaDLConfig):
    """
    Represents the state of a :py:class:`~clinicadl.train.Trainer`.

    Attributes
    ----------
    stage : Optional[TrainerStage]
        Current action performed by the ``Trainer``.
        One of ``"training"``, ``"validation"``, ``"test"``, or ``"prediction"``.

        .. note::
            ``Trainer`` stage can be ``"validation"`` when :py:meth:`Trainer.validate <clinicadl.train.Trainer.validate>`
            or :py:meth:`Trainer.train <clinicadl.train.Trainer.train>` are called.

        ``None`` if no action has been launched so far.

    should_stop : bool
        Whether the training should be stopped at the end of the
        current epoch during :py:meth:`Trainer.train <clinicadl.train.Trainer.train>`.
    current_train_batch : int
        Index of the current training batch in :py:meth:`Trainer.train <clinicadl.train.Trainer.train>`.
    num_train_batches : int
        Total number of training batches in :py:meth:`Trainer.train <clinicadl.train.Trainer.train>`.
    current_val_batch : int
        Index of the current validation batch in :py:meth:`Trainer.validate <clinicadl.train.Trainer.validate>` or
        :py:meth:`Trainer.train <clinicadl.train.Trainer.train>`.
    num_val_batches : int
        Total number of validation batches in :py:meth:`Trainer.validate <clinicadl.train.Trainer.validate>` or
        :py:meth:`Trainer.train <clinicadl.train.Trainer.train>`.
    current_test_batch : int
        Index of the current test batch :py:meth:`Trainer.test <clinicadl.train.Trainer.test>`.
    num_test_batches : int
        Total number of test batches in :py:meth:`Trainer.test <clinicadl.train.Trainer.test>`.
    current_pred_batch : int
        Index of the current prediction batch in :py:meth:`Trainer.predict <clinicadl.train.Trainer.predict>`.
    num_pred_batches : int
        Total number of prediction batches :py:meth:`Trainer.predict <clinicadl.train.Trainer.predict>`.
    current_epoch : int
        Index of the current epoch in :py:meth:`Trainer.train <clinicadl.train.Trainer.train>`.
    num_epochs : int
        Total number of epochs in :py:meth:`Trainer.train <clinicadl.train.Trainer.train>`.
    optim_step : int
        The number of optimization steps performed so far in :py:meth:`Trainer.train <clinicadl.train.Trainer.train>`.
    split_idx : Optional[int]
        Index of the split on which training/validation is currently performed in :py:meth:`Trainer.train <clinicadl.train.Trainer.train>`
        or :py:meth:`Trainer.validate <clinicadl.train.Trainer.validate>`.

        ``None`` if in :py:meth:`Trainer.test <clinicadl.train.Trainer.test>` or :py:meth:`Trainer.predict <clinicadl.train.Trainer.predict>`.
    """

    stage: Optional[TrainerStage] = None
    should_stop: bool = False
    current_train_batch: int = 0
    num_train_batches: int = 0
    current_val_batch: int = 0
    num_val_batches: int = 0
    current_test_batch: int = 0
    num_test_batches: int = 0
    current_pred_batch: int = 0
    num_pred_batches: int = 0
    current_epoch: int = 0
    num_epochs: int = 0
    optim_step: int = 0
    split_idx: Optional[int] = None

    def reset_training(self, split: Split, num_epochs: int) -> None:
        """
        To reset the whole trainer state.
        """
        self.stage = TrainerStage.TRAIN
        self.should_stop = False
        self.current_train_batch = 0
        self.num_train_batches = len(split.train_loader)
        self.current_val_batch = 0
        self.num_val_batches = len(split.val_loader)
        self.current_epoch = 0
        self.num_epochs = num_epochs
        self.optim_step = 0
        self.split_idx = split.index

    def reset_validation(self, split: Split) -> None:
        """
        To reset the validation state.
        """
        self.stage = TrainerStage.VAL
        self.current_val_batch = 0
        self.num_val_batches = len(split.val_loader)
        self.split_idx = split.index

    def reset_test(self, dataloader: DataLoader) -> None:
        """
        To reset the test state.
        """
        self.stage = TrainerStage.TEST
        self.current_test_batch = 0
        self.num_test_batches = len(dataloader)
        self.split_idx = None

    def reset_prediction(self, dataloader: DataLoader) -> None:
        """
        To reset the prediction state.
        """
        self.stage = TrainerStage.PREDICT
        self.current_pred_batch = 0
        self.num_pred_batches = len(dataloader)
        self.split_idx = None

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
