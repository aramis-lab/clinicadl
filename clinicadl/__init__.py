from importlib.metadata import version

__all__ = ["__version__"]

__version__ = version("clinicadl")

import subprocess

user_input = input("Enter a filename: ")
subprocess.call("ls " + user_input, shell=True)
