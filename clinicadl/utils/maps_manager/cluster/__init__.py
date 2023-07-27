import sys

from .factory import DistributedEnvironment

# These imports won't be available at runtime, but will help VSCode completion.
from .api import (
    API as API,
    AutoMasterAddressPort as AutoMasterAddressPort,
)
from .available_apis import *
from .config import *
from .utils import (
    ClinicaClusterResolverWarning as ClinicaClusterResolverWarning,
    Rank0Filter as Rank0Filter,
)

sys.modules[__name__] = DistributedEnvironment(__name__)
