import os
import sys
from functools import wraps
from re import sub
from typing import Optional

import torch.profiler
from packaging.version import Version


class persistent_locals(object):
    """
    Allows access to local variables of a function.
    Shamelessly stolen from
    https://stackoverflow.com/questions/9186395/python-is-there-a-way-to-get-a-local-function-variable-from-within-a-decorator
    """

    def __init__(self, func):
        self._locals = {}
        self.func = func

    def __call__(self, *args, **kwargs):
        def tracer(frame, event, arg):
            if event == "return":
                self._locals = frame.f_locals.copy()

        # tracer is activated on next call, return or exception
        sys.setprofile(tracer)
        try:
            # trace the function call
            res = self.func(*args, **kwargs)
        finally:
            # disable tracer and replace with old one
            sys.setprofile(None)
        return res

    def clear_locals(self):
        self._locals = {}

    @property
    def locals(self):
        return self._locals


if Version(torch.__version__) >= Version("1.12.0"):
    # This tensorboard_trace_handler wraps Pytorch's version. It restores a feature
    # of Kineto profiler which was lost when upgrading Pytorch from 1.11 to 1.12.
    # In the profiler, some category names were changed. But in the tensorboard
    # visualization from Kineto, those category have not been renamed accordingly.
    # This loses the dataloader step profiling. We can restore this feature by
    # renaming the category in the output trace file.

    @wraps(torch.profiler.tensorboard_trace_handler)
    def tensorboard_trace_handler(
        dir_name: str, worker_name: Optional[str] = None, use_gzip: bool = False
    ):
        handler_fn = torch.profiler.tensorboard_trace_handler(
            dir_name=dir_name,
            worker_name=worker_name,
            use_gzip=use_gzip,
        )
        handler_fn = persistent_locals(handler_fn)

        @wraps(handler_fn)
        def wrapper(prof):
            handler_fn(prof)
            file_name = handler_fn._locals["file_name"]
            dir_name = handler_fn._locals["dir_name"]
            handler_fn.clear_locals()
            with open(os.path.join(dir_name, file_name), "r+") as file:
                content = file.read()
                file.seek(0)
                file.truncate()

                # Restore profiler steps and dataloader
                new_content = sub("user_annotation", "cpu_op", content)

                # Restore runtime category
                new_content = sub("cuda_runtime", "runtime", new_content)

                file.write(new_content)

        return wrapper

else:
    tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler
