from typing import Callable

from clinicadl.data.structures import DataPoint

Transform = Callable[[DataPoint], DataPoint]
