#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
from functools import wraps
from typing import Callable

from ..config import __all__ as all_API_methods
from .base import API


env_variables_set: bool = False


def set_master_addr_port_env_variables(func):
    @wraps(func)
    def wrapper(self):
        global env_variables_set
        if not env_variables_set:
            env_variables_set = True  # must be done before actually setting the variable to prevent stackoverflow
            os.environ["MASTER_ADDR"] = self.master_address()
            os.environ["MASTER_PORT"] = str(self.port())
        return func(self)

    return wrapper


def decorate_methods(cls: API, func_to_apply: Callable) -> Callable:
    for obj_name in dir(cls):
        if obj_name in all_API_methods:
            decorated = func_to_apply(getattr(cls, obj_name))
            setattr(cls, obj_name, decorated)

    return cls


def AutoMasterAddressPort(cls: API) -> API:
    return decorate_methods(cls, func_to_apply=set_master_addr_port_env_variables)
