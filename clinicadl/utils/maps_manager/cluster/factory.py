#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import warnings
from pathlib import Path
from typing import Callable, Any

from . import (
    __builtins__,
    __cached__,
    __name__,
    __path__,
    available_apis,
    config,
)
from .api import API, AutoMasterAddressPort
from .utils import (
    ClinicaClusterResolverWarning,
    Rank0Filter,
    descriptorize,
    warning_filter,
)


def _create():
    return APIFactory("DistributedEnvironment", (sys.__class__,), {})


class APIFactory(type):
    def make_new_func(dest_name: str) -> Callable:
        @descriptorize
        def redirect() -> Any:
            with warnings.catch_warnings(record=True) as warning_list:
                api = available_apis.get_launcher_API()
                output = getattr(api, dest_name)()
            if warning_list:
                warning_filter.warn(warning_list)
            return output

        return redirect

    def __new__(cls, name, bases, methods):
        new_methods = {**methods}
        __dir__ = []
        __all__ = []

        for method_name in config.__all__:
            new_methods[method_name] = APIFactory.make_new_func(
                getattr(config, method_name).__name__
            )

        for method_name in available_apis.__all__:
            new_methods[method_name] = getattr(available_apis, method_name)
        __all__ += available_apis.__all__

        new_methods["API"] = API
        new_methods["AutoMasterAddressPort"] = AutoMasterAddressPort
        new_methods["ClinicaClusterResolverWarning"] = ClinicaClusterResolverWarning
        new_methods["Rank0Filter"] = Rank0Filter
        __all__ += [
            "API",
            "AutoMasterAddressPort",
            "ClinicaClusterResolverWarning",
            "Rank0Filter",
        ]

        new_methods["__dir__"] = lambda *args, **kwargs: __dir__
        new_methods["__file__"] = str(Path(__file__).parent)
        new_methods["__path__"] = __path__
        new_methods["__builtins__"] = __builtins__
        new_methods["__cached__"] = __cached__
        new_methods["__name__"] = __name__
        new_methods["__all__"] = __all__

        for name in new_methods.keys():
            __dir__.append(name)

        new_methods["__reduce__"] = lambda _: (_create, ())
        return super().__new__(cls, name, bases, new_methods)


DistributedEnvironment = _create()
