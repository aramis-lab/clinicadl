#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import List

from ..utils import get_first_host
from .auto_master_addr_port import AutoMasterAddressPort
from .base import API


@AutoMasterAddressPort
class SlurmAPI(API):
    priority: int = 10000
    name: str = "Slurm"

    def is_launcher(self) -> bool:
        return "SLURM_STEP_ID" in os.environ

    def rank(self) -> int:
        return int(os.environ["SLURM_PROCID"])

    def local_rank(self) -> int:
        return int(os.environ["SLURM_LOCALID"])

    def world_size(self) -> int:
        return int(os.environ["SLURM_STEP_NUM_TASKS"])

    def local_world_size(self) -> int:
        return int(os.environ["SLURM_STEP_TASKS_PER_NODE"])

    def num_nodes(self) -> int:
        return int(os.environ["SLURM_STEP_NUM_NODES"])

    def cpus(self) -> int:
        cpu = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
        return cpu or len(os.sched_getaffinity(0))

    def gpus(self) -> List[str]:
        step_gpus = os.environ.get("SLURM_STEP_GPUS", None)
        if step_gpus is not None:
            return step_gpus.split(",")
        return []

    def nodelist(self) -> str:
        return os.environ["SLURM_STEP_NODELIST"]

    def master_address(self) -> str:
        return get_first_host(self.nodelist())

    def jobid(self) -> int:
        return int(os.environ["SLURM_JOB_ID"])

    def port(self) -> int:
        return 10000 + self.jobid() % 20000
