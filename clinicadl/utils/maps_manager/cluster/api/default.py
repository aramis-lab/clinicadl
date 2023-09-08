#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import socket
from contextlib import closing
from typing import List, Optional

from .auto_master_addr_port import AutoMasterAddressPort
from .base import API


@AutoMasterAddressPort
class DefaultAPI(API):
    priority: int = 0
    name: str = "Sequential"

    def __init__(self):
        self.current_port: Optional[int] = None

    @staticmethod
    def find_available_port() -> int:
        """
        Tries to bind to local port until it finds one which is available.
        This is used to set the master port environment variable.
        """
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.bind(("localhost", 0))
            port = sock.getsockname()[1]
        return port

    def is_launcher(self) -> bool:
        return True

    def rank(self) -> int:
        return 0

    def local_rank(self) -> int:
        return 0

    def world_size(self) -> int:
        return 1

    def local_world_size(self) -> int:
        return 1

    def num_nodes(self) -> int:
        return 1

    def cpus(self) -> int:
        return len(os.sched_getaffinity(0))

    def gpus(self) -> List[str]:
        return []

    def nodelist(self) -> List[str]:
        return ["localhost"]

    def master_address(self) -> str:
        return "localhost"

    def port(self) -> int:
        if self.current_port is None:
            self.current_port = self.find_available_port()
        return self.current_port
