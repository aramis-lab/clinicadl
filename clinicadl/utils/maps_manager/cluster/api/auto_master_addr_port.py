#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
from functools import wraps
from typing import Callable, Type

from ..config import __all__ as all_API_methods
from .base import API

# Defines a class decorator to make wraps any API methods so that the Master Address
# and the Master Port are set in order to allow the process group to initialize
# correctly.

env_variables_set: bool = False


def set_master_addr_port_env_variables(func):
    # The parameter should be a method of a subclass of the API abstract class.
    @wraps(func)
    def wrapper(self):
        global env_variables_set
        if not env_variables_set:
            env_variables_set = True  # must be done before actually setting the variable to prevent stackoverflow
            os.environ["MASTER_ADDR"] = self.master_address()
            os.environ["MASTER_PORT"] = str(self.port())
        return func(self)

    return wrapper


def decorate_methods(cls: Type[API], func_to_apply: Callable) -> Type[API]:
    # Decorate all API methods defined in the config file with the given function.
    for obj_name in dir(cls):
        if obj_name in all_API_methods:
            decorated = func_to_apply(getattr(cls, obj_name))
            setattr(cls, obj_name, decorated)

    return cls


def AutoMasterAddressPort(cls: Type[API]) -> Type[API]:
    # When we call a cluster API function for the first time, we set the MASTER_ADDR
    # and MASTER_PORT environment variables, so that the Pytorch wrapper
    # DistributedDataParallel can set up communication correctly.
    return decorate_methods(cls, func_to_apply=set_master_addr_port_env_variables)
