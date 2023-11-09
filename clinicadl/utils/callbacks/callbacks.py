from logging import getLogger

logger = getLogger("clinicadl.callbacks")


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

    def __init__(self):
        self.callbacks = []
        # for cb in callbacks:
        #     self.add_callback(cb)
        # self.model = model

    def add_callback(self, callback):
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__
        if cb_class in [c.__class__ for c in self.callbacks]:
            logger.warning(
                f"You are adding a {cb_class} to the callbacks but there one is already used."
                f" The current list of callbacks is\n: {self.callback_list}"
            )
        self.callbacks.append(cb)

    @property
    def callback_list(self):
        return "\n".join(cb.__class__.__name__ for cb in self.callbacks)

    def on_train_begin(self, data_config, **kwargs):
        self.call_event("on_train_begin", data_config, **kwargs)

    def on_train_end(self, data_config, **kwargs):
        self.call_event("on_train_end", data_config, **kwargs)

    def on_epoch_begin(self, data_config, **kwargs):
        self.call_event("on_epoch_begin", data_config, **kwargs)

    def on_epoch_end(self, data_config, **kwargs):
        self.call_event("on_epoch_end", data_config, **kwargs)

    def on_batch_begin(self, data_config, **kwargs):
        self.call_event("on_batch_begin", data_config, **kwargs)

    def on_batch_end(self, data_config, **kwargs):
        self.call_event("on_batch_end", data_config, **kwargs)

    def on_loss_begin(self, data_config, **kwargs):
        self.call_event("on_loss_begin", data_config, **kwargs)

    def on_loss_end(self, data_config, **kwargs):
        self.call_event("on_loss_end", data_config, **kwargs)

    def on_step_begin(self, data_config, **kwargs):
        self.call_event("on_step_begin", data_config, **kwargs)

    def on_step_end(self, data_config, **kwargs):
        self.call_event("on_step_end", data_config, **kwargs)

    def call_event(self, event, data_config, **kwargs):
        for callback in self.callbacks:
            result = getattr(callback, event)(
                data_config,
                # model=self.model,
                **kwargs,
            )


# class LearningRateScheduler(Callback):
#     def on_train_begin(self, codecarbon_bool, **kwargs):
#         # control the learning rate over iteration
#         # self.optimizer.lr = fct(iteration)
#         print("test réussi")

#     def on_train_end(self, codecarbon_bool, **kwargs):
#         print("ok")


class CodeCarbonTracker(Callback):
    def on_train_begin(self, data_config, **kwargs):
        from codecarbon import EmissionsTracker

        # my_logger = LoggerOutput(logger, logging.WARNING)
        self.tracker = EmissionsTracker()
        self.tracker.start()

    def on_train_end(self, data_config, **kwargs):
        self.tracker.stop()
