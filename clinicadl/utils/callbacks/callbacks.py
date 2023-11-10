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

    def on_train_begin(self, parameters, **kwargs):
        self.call_event("on_train_begin", parameters, **kwargs)

    def on_train_end(self, parameters, **kwargs):
        self.call_event("on_train_end", parameters, **kwargs)

    def on_epoch_begin(self, parameters, **kwargs):
        self.call_event("on_epoch_begin", parameters, **kwargs)

    def on_epoch_end(self, parameters, **kwargs):
        self.call_event("on_epoch_end", parameters, **kwargs)

    def on_batch_begin(self, parameters, **kwargs):
        self.call_event("on_batch_begin", parameters, **kwargs)

    def on_batch_end(self, parameters, **kwargs):
        self.call_event("on_batch_end", parameters, **kwargs)

    def on_loss_begin(self, parameters, **kwargs):
        self.call_event("on_loss_begin", parameters, **kwargs)

    def on_loss_end(self, parameters, **kwargs):
        self.call_event("on_loss_end", parameters, **kwargs)

    def on_step_begin(self, parameters, **kwargs):
        self.call_event("on_step_begin", parameters, **kwargs)

    def on_step_end(self, parameters, **kwargs):
        self.call_event("on_step_end", parameters, **kwargs)

    def call_event(self, event, parameters, **kwargs):
        for callback in self.callbacks:
            result = getattr(callback, event)(
                parameters,
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
    def on_train_begin(self, parameters, **kwargs):
        from codecarbon import EmissionsTracker

        # my_logger = LoggerOutput(logger, logging.WARNING)
        self.tracker = EmissionsTracker()
        self.tracker.start()

    def on_train_end(self, parameters, **kwargs):
        self.tracker.stop()


class Tracker(Callback):
    def on_train_begin(self, parameters, **kwargs):
        if parameters["track_exp"] == "wandb":
            from clinicadl.utils.tracking_exp import WandB_handler

            self.run = WandB_handler(
                kwargs["split"], parameters, kwargs["maps_path"].name
            )

        if parameters["track_exp"] == "mlflow":
            from clinicadl.utils.tracking_exp import Mlflow_handler

            self.run = Mlflow_handler(
                kwargs["split"], parameters, kwargs["maps_path"].name
            )

    def on_epoch_end(self, parameters, **kwargs):
        if parameters["track_exp"] == "wandb":
            self.run.log_metrics(
                self.run._wandb,
                parameters["track_exp"],
                parameters["network_task"],
                kwargs["metrics_train"],
                kwargs["metrics_valid"],
            )

        if parameters["track_exp"] == "mlflow":
            self.run.log_metrics(
                self.run._mlflow,
                parameters["track_exp"],
                parameters["network_task"],
                kwargs["metrics_train"],
                kwargs["metrics_valid"],
            )

    def on_train_end(self, parameters, **kwargs):
        if parameters["track_exp"] == "mlflow":
            self.run._mlflow.end_run()

        if parameters["track_exp"] == "wandb":
            self.run._wandb.finish()


class LoggerCallback(Callback):
    def on_train_begin(self, parameters, **kwargs):
        logger.info(
            f"Criterion for {parameters['network_task']} is {(kwargs['criterion'])}"
        )
        logger.debug(f"Optimizer used for training is {kwargs['optimizer']}")

    def on_epoch_begin(self, parameters, **kwargs):
        logger.info(f"Beginning epoch {kwargs['epoch']}.")

    def on_epoch_end(self, parameters, **kwargs):
        logger.info(
            f"{kwargs['mode']} level training loss is {kwargs['metrics_train']['loss']} "
            f"at the end of iteration {kwargs['i']}"
        )
        logger.info(
            f"{kwargs['mode']} level validation loss is {kwargs['metrics_valid']['loss']} "
            f"at the end of iteration {kwargs['i']}"
        )

    def on_train_end(self, parameters, **kwargs):
        logger.info("tests")


# class ProfilerHandler(Callback):
#     def on_train_begin(self, parameters, **kwargs):
#         if self.profiler:
#             from contextlib import nullcontext
#             from datetime import datetime
#             from clinicadl.utils.maps_manager.cluster.profiler import (
#                 ProfilerActivity,
#                 profile,
#                 schedule,
#                 tensorboard_trace_handler,
#             )

#             time = datetime.now().strftime("%H:%M:%S")
#             filename = [self.maps_path / "profiler" / f"clinicadl_{time}"]
#             dist.broadcast_object_list(filename, src=0)
#             profiler = profile(
#                 activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#                 schedule=schedule(wait=2, warmup=2, active=30, repeat=1),
#                 on_trace_ready=tensorboard_trace_handler(filename[0]),
#                 profile_memory=True,
#                 record_shapes=False,
#                 with_stack=False,
#                 with_flops=False,
#             )
#         else:
#             profiler = nullcontext()
#             profiler.step = lambda *args, **kwargs: None
#         return profiler
