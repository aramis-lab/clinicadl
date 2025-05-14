class Callback:
    def __init__(self):
        pass

    def on_train_begin(self, **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass

    def on_epoch_begin(self, epoch: int, **kwargs):
        pass

    def on_epoch_end(self, epoch: int, **kwargs):
        pass

    def on_batch_begin(self, batch: int, **kwargs):
        pass

    def on_batch_end(self, batch: int, **kwargs):
        pass

    def on_loss_begin(self, **kwargs):
        pass

    def on_loss_end(self, **kwargs):
        pass

    def on_step_begin(self, **kwargs):
        pass

    def on_step_end(self, **kwargs):
        pass
