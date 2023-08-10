#! /usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
from inspect import isclass
from pathlib import Path
from typing import Any, Callable, List

from . import __cached__, __name__, __path__
from .api import API, AutoMasterAddressPort
from .utils import (
    ClinicaClusterResolverWarning,
    Rank0Filter,
    warning_filter,
)


class EmptyClass(object):
    pass


class Interface(object):

    def __init__(self):
        self._available_APIs: List[API] = []
        self.crawl_shipped_APIs()
        self.add_other_object_for_easy_access()
        self.add_API_functions()
        self.make_dir()

    @classmethod
    def add_attribute(cls, name, attribute) -> None:
        setattr(cls, name, attribute)

    def add_API_functions(self) -> None:
        from . import config
        for method_name in config.__all__:
            self.add_attribute(method_name, self.make_new_func(
                getattr(config, method_name).__name__
            ))

    def make_dir(self):
        from . import config
        self.__dir: List[str] = []
        self.__dir += dir(EmptyClass())
        self.__dir += config.__all__
        self.__dir += self.__all__

    def crawl_shipped_APIs(self) -> None:
        from . import api
        self.crawl_module_for_APIs(api)

    def add_other_object_for_easy_access(self) -> None:
        from . import api
        self.api = api
        self.API = API
        self.AutoMasterAddressPort = AutoMasterAddressPort
        self.ClinicaClusterResolverWarning = ClinicaClusterResolverWarning
        self.Rank0Filter = Rank0Filter
        self.__file__ = str(Path(__file__).parent)
        self.__path__ = __path__
        self.__cached__ = __cached__
        self.__name__ = __name__
        self.__all__ = [
            "api",
            "API",
            "AutoMasterAddressPort",
            "ClinicaClusterResolverWarning",
            "Rank0Filter",
            "register_API",
            "get_launcher_API",
            "current_API",
            "all_APIs",
            "crawl_module_for_APIs",
            "__file__",
            "__path__",
            "__cached__",
            "__name__",
        ]

    def __dir__(self) -> str:
        return self.__dir

    def make_new_func(self, dest_name: str) -> Callable:
        def redirect(self: Interface) -> Any:
            with warnings.catch_warnings(record=True) as warning_list:
                api = self.get_launcher_API()
                output = getattr(api, dest_name)()
            if warning_list:
                warning_filter.warn(warning_list)
            return output
        return property(redirect)

    def register_API(self, new_API: API) -> None:
        for i, api in enumerate(self._available_APIs):
            if api.priority > new_API.priority:
                continue
            else:
                self._available_APIs.insert(i, new_API)
                break
        else:
            self._available_APIs.append(new_API)

    def get_launcher_API(self) -> API:
        for api in self._available_APIs:
            if api.is_launcher():
                return api

    @property
    def current_API(self) -> str:
        return self.get_launcher_API().name

    @property
    def all_APIs(self) -> List[API]:
        return self._available_APIs

    def crawl_module_for_APIs(self, module) -> None:
        for obj_name in dir(module):
            obj = getattr(module, obj_name)
            if isclass(obj) and issubclass(obj, API) and obj is not API:
                # obj is the class so we instanciate it
                self.register_API(obj())
            elif isinstance(obj, API) and obj.__class__ is not API:
                # obj is already the instance
                self.register_API(obj)
