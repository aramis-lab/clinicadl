import sys

from .interface import Interface

# These imports won't be available at runtime, but will help VSCode completion.
from .api import API as API
from .api import AutoMasterAddressPort as AutoMasterAddressPort
from .config import *
from .utils import ClinicaClusterResolverWarning as ClinicaClusterResolverWarning
from .utils import Rank0Filter as Rank0Filter

sys.modules[__name__] = Interface()
