from importlib.metadata import version

from .utils.maps_manager import MapsManager

__all__ = ["__version__", "MapsManager"]

__version__ = version("clinicadl")
