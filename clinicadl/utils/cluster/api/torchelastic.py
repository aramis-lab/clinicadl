#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import warnings
from typing import List

from ..utils import ClinicaClusterResolverWarning
from .base import API


class TorchElasticAPI(API):
    priority: int = 9000
    name: str = "TorchElastic"

    def is_launcher(self) -> bool:
        return "TORCHELASTIC_RUN_ID" in os.environ

    def rank(self) -> int:
        return int(os.environ["RANK"])

    def local_rank(self) -> int:
        return int(os.environ["LOCAL_RANK"])

    def world_size(self) -> int:
        return int(os.environ["WORLD_SIZE"])

    def local_world_size(self) -> int:
        return int(os.environ["LOCAL_WORLD_SIZE"])

    def num_nodes(self) -> int:
        return self.world_size() // self.local_world_size()

    def cpus(self) -> int:
        return len(os.sched_getaffinity(0)) // self.local_world_size()

    def gpus(self) -> List[str]:
        return [str(i) for i in range(self.local_world_size())]

    def nodelist(self) -> List[str]:
        warnings.warn(
            "TorchElastic does not provide the whole list of nodes "
            "involved in a distributed computation. So you will only "
            "get the address of the master.",
            category=ClinicaClusterResolverWarning,
            stacklevel=4,
        )
        return [self.master_address()]

    def master_address(self) -> str:
        # Torchrun already defines the Master address.
        return os.environ["MASTER_ADDR"]

    def port(self) -> int:
        # Torchrun already defines the Master port.
        return int(os.environ["MASTER_PORT"])
