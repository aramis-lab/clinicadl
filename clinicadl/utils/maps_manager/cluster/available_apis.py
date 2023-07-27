#! /usr/bin/env python
# -*- coding: utf-8 -*-

from inspect import isclass
from typing import List

from . import api
from .api import API
from .utils import descriptorize


__all__ = [
    "all_APIs",
    "current_API",
    "crawl_module_for_APIs",
    "get_APIs",
    "register_API",
]

available_APIs: List[API] = []


def register_API(new_API: API) -> None:
    for i, api in enumerate(available_APIs):
        if api.priority > new_API.priority:
            continue
        else:
            available_APIs.insert(i, new_API)
            break
    else:
        available_APIs.append(new_API)


def get_APIs() -> List[API]:
    return available_APIs


def get_launcher_API() -> API:
    for api in get_APIs():
        if api.is_launcher():
            return api


def crawl_module_for_APIs(module) -> None:
    for obj_name in dir(module):
        obj = getattr(module, obj_name)
        if isclass(obj) and issubclass(obj, API) and obj is not API:
            # obj is the class so we instanciate it
            register_API(obj())
        elif isinstance(obj, API) and obj.__class__ is not API:
            # obj is already the instance
            register_API(obj)


current_API = descriptorize(lambda: get_launcher_API().name)
all_APIs = descriptorize(lambda: [api.name for api in get_APIs()].__repr__())


crawl_module_for_APIs(api)
