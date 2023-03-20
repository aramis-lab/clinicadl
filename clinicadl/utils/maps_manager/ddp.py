import os
from abc import ABC, abstractmethod
from logging import Logger
from typing import Optional

import torch
import torch.distributed as dist

from clinicadl.utils.logger import Rank0Filter


class ClusterResolver(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def rank(self) -> int:
        """
        Returns the rank of the given process within the communicator
        """
        pass

    @property
    @abstractmethod
    def world_size(self) -> int:
        """
        Returns the number of processes in the communicator
        """
        pass

    @property
    @abstractmethod
    def local_rank(self) -> int:
        """
        Returns the rank of the given process within the node
        """
        pass

    @property
    @abstractmethod
    def master(self) -> bool:
        """
        Returns whether or not the given process is considered the master
        """
        pass

    @property
    @abstractmethod
    def master_addr(self) -> str:
        """
        Returns the address of the master for Pytorch to setup distribution
        """
        pass

    @property
    @abstractmethod
    def master_port(self) -> int:
        """
        Returns the port on the master node (should avoid port conflict)
        """
        pass


class MonoTaskResolver(ClusterResolver):
    @property
    def rank(self) -> int:
        return 0

    @property
    def world_size(self) -> int:
        return 1

    @property
    def local_rank(self) -> int:
        return 0

    @property
    def master(self) -> bool:
        return True

    @property
    def master_addr(self) -> str:
        return "127.0.0.1"

    @property
    def master_port(self) -> int:
        return 12345


class SlurmClusterResolver(ClusterResolver):
    def __init__(self):
        super().__init__()
        self.reference_port = 12345

    @property
    def rank(self) -> int:
        return int(os.environ["SLURM_PROCID"])

    @property
    def world_size(self) -> int:
        return int(os.environ["SLURM_NTASKS"])

    @property
    def local_rank(self) -> int:
        return int(os.environ["SLURM_LOCALID"])

    @property
    def master(self) -> bool:
        return self.rank == 0

    @staticmethod
    def get_first_host(hostlist: str) -> str:
        from re import findall, split, sub

        regex = "\[([^[\]]*)\]"
        all_replacement: list[str] = findall(regex, hostlist)
        new_values = [split("-|,", element)[0] for element in all_replacement]
        for i in range(len(new_values)):
            hostlist = sub(regex, new_values[i], hostlist, count=1)
        return hostlist.split(",")[0]

    @property
    def master_addr(self) -> str:
        return self.get_first_host(os.environ["SLURM_JOB_NODELIST"])

    @property
    def master_port(self) -> int:
        return self.reference_port + int(min(os.environ["SLURM_STEP_GPUS"].split(",")))


class DDPManager:
    def __init__(
        self,
        ddp: bool,
        resolver: str,
        gpu: bool = True,
        logger: Optional[Logger] = None,
    ):
        global ddp_manager
        assert ddp_manager is None, "You cannot initialize DDP twice."
        self.ddp = ddp
        self.gpu = gpu
        self.resolver: ClusterResolver = self.get_resolver(ddp, resolver)
        self.init_ddp()
        if logger is not None:
            logger.addFilter(Rank0Filter(rank=self.rank))

    def init_ddp(self):
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        dist.init_process_group(
            backend="nccl" if self.gpu else "gloo",
            init_method="env://",
            world_size=self.world_size,
            rank=self.rank,
        )
        if self.gpu:
            torch.cuda.set_device(self.local_rank)

    @staticmethod
    def get_resolver(ddp: bool, resolver: str) -> ClusterResolver:
        if not ddp:
            return MonoTaskResolver()
        if resolver.lower() == "slurm":
            return SlurmClusterResolver()
        else:
            raise NotImplementedError("This resolver has not been implemented yet")

    @property
    def rank(self) -> int:
        return self.resolver.rank

    @property
    def world_size(self) -> int:
        return self.resolver.world_size

    @property
    def local_rank(self) -> int:
        return self.resolver.local_rank

    @property
    def master(self) -> bool:
        return self.resolver.master

    @property
    def master_addr(self) -> str:
        return self.resolver.master_addr

    @property
    def master_port(self) -> int:
        return self.resolver.master_port


ddp_manager: Optional[DDPManager] = None


def get_ddp_manager(*args, **kwargs):
    global ddp_manager
    if ddp_manager is None:
        ddp_manager = DDPManager(*args, **kwargs)
    return ddp_manager
