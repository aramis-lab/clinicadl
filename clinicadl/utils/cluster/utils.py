#! /usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
from logging import Filter
from re import findall, split, sub
from typing import List, Set, Tuple, Type, Union


def get_first_host(hostlist: str) -> str:
    """
    Get the first host from SLURM's nodelist.
    Example: Nodelist="Node[1-5],Node7" -> First node: "Node1"

    Args:
        hostlist(str): the compact nodelist as given by SLURM
    Returns:
        (str): the first node to host the master process
    """
    regex = r"\[([^[\]]*)\]"
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
    """
    Custom warning filter to make sure our own warnings are only sent once.
    Also solves the stacklevel issue from the cluster resolver.
    """

    def __init__(self):
        self.registry: Set[Tuple[str, Union[Type[str], Type[Warning]]]] = set()

    def block(self, warning: Union[str, Warning]) -> bool:
        """
        Checks whether or not a warning should be intercepted.
        """
        text = str(warning)
        category = warning.__class__
        if not isinstance(warning, ClinicaClusterResolverWarning):
            return False
        message = (text, category)
        if message not in self.registry:
            self.registry.add(message)
            return False
        return True

    def warn(self, warning_list: List[warnings.WarningMessage]):
        for warning in warning_list:
            if not self.block(warning.message):
                category = (
                    warning.message.__class__
                    if isinstance(warning.message, Warning)
                    else Warning
                )
                warnings.warn(
                    message=str(warning.message),
                    category=category,
                    stacklevel=3,
                )


warning_filter = WarningFilter()
