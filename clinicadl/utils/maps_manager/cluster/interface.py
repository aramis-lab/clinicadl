#! /usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
from inspect import isclass
from pathlib import Path
from typing import Any, Iterable, List

from . import __name__, __path__
from .api import API, AutoMasterAddressPort, DefaultAPI
from .utils import ClinicaClusterResolverWarning, Rank0Filter, warning_filter


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
        # Our interface should have all functions defined in the config file.
        # We use a custom wrapper to determine which API should be used.
        from . import config

        for method_name in config.__all__:
            self.add_attribute(
                method_name, self.make_new_func(getattr(config, method_name).__name__)
            )

    def make_dir(self):
        # Set the __dir__ method so that we can find all API functions as well as base classes attributes.
        from . import config

        self.__dir: List[str] = []
        self.__dir += dir(EmptyClass())
        self.__dir += config.__all__
        self.__dir += self.__all__

    def crawl_shipped_APIs(self) -> None:
        # Every API defined in the api folder should directly be added so that
        # the interface is aware of its existence. This mechanism allows a user
        # to define a custom API and then add it by himself.
        from . import api

        self.crawl_module_for_APIs(api)

    def add_other_object_for_easy_access(self) -> None:
        # For convenience purposes, we may need to access a few method we defined
        # on top of cluster API functions. It may be useful for instance if a user
        # wants to define a new API.
        from . import api

        self.api = api
        self.API = API
        self.AutoMasterAddressPort = AutoMasterAddressPort
        self.ClinicaClusterResolverWarning = ClinicaClusterResolverWarning
        self.Rank0Filter = Rank0Filter
        self.__file__ = str(Path(__file__).parent)
        self.__path__ = __path__
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
        ]

    def __dir__(self) -> Iterable[str]:
        return self.__dir

    def make_new_func(self, dest_name: str) -> property:
        # When a user will call an API method, we need to find out which API it is
        # appropriate to use. Every API should implement a method to check whether
        # it is active or not. Then we use this API's method to return the result
        # to the user.
        #
        # This check is performed at runtime because the cluster API may change at
        # runtime. This is for instance the case if we use Meta's submitit library.
        # In this situation, we import the cluster library in a single process
        # on a front node. Then the code is executed within a MPI environment.
        #
        # If a warning is raised by any API, we make sure that it is only raised once,
        # as well as correct the stacklevel so that it corresponds to the actual
        # line of code from the user.

        def redirect(self: Interface) -> Any:
            with warnings.catch_warnings(record=True) as warning_list:
                api = self.get_launcher_API()
                output = getattr(api, dest_name)()
            if warning_list:
                warning_filter.warn(warning_list)
            return output

        return property(redirect)

    def register_API(self, new_API: API) -> None:
        """
        Add a new API to the register so that it can be used later on.
        They are sorted according to their priority. Higher priority APIs are checked
        first. If a API assure that it is activated, lower priority APIs will not even
        be checked. The Default API (mono-process, always say that it is activated)
        has priority 0.

        Args:
            new_API (API): The new API to add to the list.
        """
        for i, api in enumerate(self._available_APIs):
            if api.priority > new_API.priority:
                continue
            else:
                self._available_APIs.insert(i, new_API)
                break
        else:
            self._available_APIs.append(new_API)

    def get_launcher_API(self) -> API:
        """
        Find which API is currently activated. They are checked in decreasing order of
        priority. If an API is activated, lower priority APIs are not checked.
        Default API (mono-process) has priority 0.

        Returns:
            API: the API which is currently used to perform parallelism.
        """
        for api in self._available_APIs:
            if api.is_launcher():
                return api
        # This should never trigger but it is a safety precaution.
        # It also helps mypy understand that this code is in fact correct.
        return DefaultAPI()

    @property
    def current_API(self) -> str:
        # Gets the name of the currently activated API.
        return self.get_launcher_API().name

    @property
    def all_APIs(self) -> List[API]:
        # Returns all APIs, for debugging purposes.
        return self._available_APIs

    def crawl_module_for_APIs(self, module) -> None:
        """
        We want to automatically add any API which is found in a given module. This
        is used to add APIs which are shipped with the cluster package, in the api
        subpackage.

        If a user wants to add another freshly implemented API, it does not need to
        copy it within clinicadl install directory, it can instead use the register_API
        method. If multiple API are implemented in a module, the module can directly
        be given to the crawl_module_for_APIs method.

        The module needs to implement the __dir__ method.

        Args:
            module (Any): Module which will be crawled to find any acceptable API object.
        """
        for obj_name in dir(module):
            obj = getattr(module, obj_name)
            if isclass(obj) and issubclass(obj, API) and obj is not API:
                # obj is the class so we instanciate it
                self.register_API(obj())
            elif isinstance(obj, API) and obj.__class__ is not API:
                # obj is already the instance
                self.register_API(obj)
