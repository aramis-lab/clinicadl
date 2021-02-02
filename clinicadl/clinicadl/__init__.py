__all__ = ['__version__']

# Load the Clinica package version
import sys
import pkgutil
__version__ = pkgutil.get_data(__package__, 'VERSION').decode('ascii').strip()
version = __version__

# import pkg_resources
# version = pkg_resources.require("Clinica")[0].version
# __version__ = version

# python 3.6 minimum version is required
if sys.version_info < (3, 6):
    print(f"ClinicaDL {__version__} requires Python 3.6")
    sys.exit(1)
