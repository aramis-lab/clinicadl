from clinicadl.utils.config.training import _TrainingState


class Callback:
    """Base class for callbacks."""

    def __init__(self):
        pass

    def on_train_begin(self, config: _TrainingState, **kwargs):
        pass

    def on_train_end(self, config: _TrainingState, **kwargs):
        pass

    def on_epoch_begin(self, config: _TrainingState, **kwargs):
        pass

    def on_epoch_end(self, config: _TrainingState, **kwargs):
        pass

    def on_batch_begin(self, config: _TrainingState, **kwargs):
        pass

    def on_batch_end(self, config: _TrainingState, **kwargs):
        pass

    def on_backward_begin(self, config: _TrainingState, **kwargs):
        pass

    def on_backward_end(self, config: _TrainingState, **kwargs):
        pass

    def on_validation_begin(self, config: _TrainingState, **kwargs):
        pass

    def on_validation_end(self, config: _TrainingState, **kwargs):
        pass
