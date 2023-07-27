#! /usr/bin/env python
# -*- coding: utf-8 -*-

from logging import Filter
from re import findall, split, sub
from typing import Callable, List, Tuple, Set
import warnings


def titlecase(string: str) -> str:
    return "".join(x for x in string.title() if x.isalnum())


def descriptorize(getter: Callable):
    descriptor_cls = type(
        titlecase(getter.__name__),
        (),
        {
            "__get__": lambda *args, **kwargs: getter(),
        },
    )
    return descriptor_cls()


def get_first_host(hostlist: str) -> str:
    regex = "\[([^[\]]*)\]"
    all_replacement: list[str] = findall(regex, hostlist)
    new_values = [split("-|,", element)[0] for element in all_replacement]
    for i in range(len(new_values)):
        hostlist = sub(regex, new_values[i], hostlist, count=1)
    return hostlist.split(",")[0]


class Rank0Filter(Filter):  # only log if it's rank 0 to avoid mess
    def __init__(self, rank: int = 0):
        super().__init__()
        self.rank = rank

    def filter(self, record):
        return self.rank == 0


class ClinicaClusterResolverWarning(RuntimeWarning):
    """
    Type (subtype of RuntimeWarning) of all warnings raised by the cluster resolver.
    You can use it to customize warning filters.
    """

    pass


class WarningFilter:

    def __init__(self):
        self.registry: Set[Tuple[str, type[ClinicaClusterResolverWarning]]] = set()

    def block(self, warning: Warning) -> bool:
        text = str(warning)
        category = warning.__class__
        if not isinstance(warning, ClinicaClusterResolverWarning):
            return False
        message = (text, category)
        if message not in self.registry:
            self.registry.add(message)
            return False
        return True

    def warn(self, warning_list: List[Warning]):
        for warning in warning_list:
            if not self.block(warning.message):
                warnings.warn(
                    message=str(warning.message),
                    category=warning.message.__class__,
                    stacklevel=4,
                )


warning_filter = WarningFilter()
