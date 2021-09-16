from .utils.maps_manager import MapsManager

__all__ = ["__version__", "MapsManager"]

# Load the Clinica package version
import pkgutil
import sys

__version__ = pkgutil.get_data(__package__, "VERSION").decode("ascii").strip()
version = __version__

# import pkg_resources
# version = pkg_resources.require("Clinica")[0].version
# __version__ = version

# python 3.6 minimum version is required
if sys.version_info < (3, 7):
    print(f"ClinicaDL {__version__} requires Python 3.7")
    sys.exit(1)
