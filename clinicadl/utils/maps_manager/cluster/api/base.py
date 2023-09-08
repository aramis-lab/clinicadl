#! /usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import List, Union


class API(ABC):
    priority: int = 5000
    name: str = "AbstractAPI"

    @abstractmethod
    def is_launcher(self) -> bool:
        """
        Detects if the given API is the one used to launch the current job.
        """
        raise NotImplementedError()

    @abstractmethod
    def rank(self) -> int:
        """
        Property containing the rank of the process.
        """
        raise NotImplementedError()

    @abstractmethod
    def local_rank(self) -> int:
        """
        Property containing the local rank of the process.
        """
        raise NotImplementedError()

    @abstractmethod
    def world_size(self) -> int:
        """
        Property containing the number of processes launched.
        """
        raise NotImplementedError()

    @abstractmethod
    def local_world_size(self) -> int:
        """
        Property containing the number of processes launched of each node.
        """
        raise NotImplementedError()

    @abstractmethod
    def num_nodes(self) -> int:
        """
        Property containing the number of nodes.
        """
        raise NotImplementedError()

    @abstractmethod
    def cpus(self) -> int:
        """
        Property containing the number of CPUs allocated to each process.
        """
        raise NotImplementedError()

    @abstractmethod
    def gpus(self) -> List[str]:
        """
        Property containing all GPUs ids.
        """
        raise NotImplementedError()

    @abstractmethod
    def nodelist(self) -> Union[str, List[str]]:
        """
        Property containing the list of nodes.
        """
        raise NotImplementedError()

    @abstractmethod
    def master_address(self) -> str:
        """
        Property containing the master node.
        """
        raise NotImplementedError()

    @abstractmethod
    def port(self) -> int:
        """
        Property containing the port to communicate with the master process.
        """
        raise NotImplementedError()

    def is_master(self) -> bool:
        """
        Detects whether or not the given process is the master (i.e. rank 0)
        """
        return self.rank() == 0
