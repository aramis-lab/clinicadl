from typing import Optional, Union


class Callback:
    def __init__(self):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass

    def on_batch_begin(self):
        pass

    def on_batch_end(self):
        pass

    def on_loss_begin(self):
        pass

    def on_loss_end(self):
        pass

    def on_step_begin(self):
        pass

    def on_step_end(self):
        pass


class CallbacksHandler:
    """
    Class to handle list of Callback.
    """

    def __init__(
        self,
        mlflow: bool = False,
        wandb: bool = False,
        codecarbon: bool = True,
        logger: bool = True,
        custom_callback: Optional[Union[Callback, list[Callback]]] = None,
    ):
        self.callbacks = []
        if codecarbon:
            self.callbacks.append(CodeCarbonCallback())
        if logger:
            self.callbacks.append(LoggerCallback())
        if mlflow:
            self.callbacks.append(MLFLOWCallback())
        if wandb:
            self.callbacks.append(WandBCallback())

        if custom_callback is not None:
            if isinstance(custom_callback, list):
                for cb in custom_callback:
                    self.callbacks.append(cb)
            elif isinstance(custom_callback, Callback):
                self.callbacks.append(custom_callback)
            else:
                raise TypeError(
                    f"Custom callback should be of type {Callback} or list of {Callback}, got {type(custom_callback)}"
                )

    def add_callback(self, callback: Callback):
        """TO COMPLETE"""

        if not isinstance(callback, Callback):
            raise TypeError(
                f"Callback should be of type {Callback}, got {type(callback)}"
            )

        if callback not in self.callbacks:
            self.callbacks.append(callback)

    @property
    def callback_list(self):
        return [cb.__class__.__name__ for cb in self.callbacks]

    def on_train_begin(self, **kwargs):
        self.call_event("on_train_begin", **kwargs)

    def on_train_end(self, **kwargs):
        self.call_event("on_train_end", **kwargs)

    def on_epoch_begin(self, **kwargs):
        self.call_event("on_epoch_begin", **kwargs)

    def on_epoch_end(self, **kwargs):
        self.call_event("on_epoch_end", **kwargs)

    def on_batch_begin(self, **kwargs):
        self.call_event("on_batch_begin", **kwargs)

    def on_batch_end(self, **kwargs):
        self.call_event("on_batch_end", **kwargs)

    def on_loss_begin(self, **kwargs):
        self.call_event("on_loss_begin", **kwargs)

    def on_loss_end(self, **kwargs):
        self.call_event("on_loss_end", **kwargs)

    def on_step_begin(self, **kwargs):
        self.call_event("on_step_begin", **kwargs)

    def on_step_end(self, **kwargs):
        self.call_event("on_step_end", **kwargs)

    def call_event(self, event, **kwargs):
        for callback in self.callbacks:
            result = getattr(callback, event)(
                **kwargs,
            )


class CodeCarbonCallback(Callback):
    def on_train_begin(self, **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass


class LoggerCallback(Callback):
    def on_train_begin(self, **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass


class WandBCallback(Callback):
    def on_train_begin(self, **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass


class MLFLOWCallback(Callback):
    def on_train_begin(self, **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass


# TODO: add WandB, MLFLOW, CodeCarbon, Tensorboard, LearningRateScheduler, EarlyStopping, ModelCheckpoint etc...
