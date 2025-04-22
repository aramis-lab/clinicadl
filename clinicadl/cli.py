import argparse

from clinicadl import __version__


def main():
    parser = argparse.ArgumentParser(
        prog="clinicadl",
        description="ClinicaDL - Deep learning Library for neuroimaging analysis.",
        epilog=(
            "For more information, visit the documentation at "
            "https://clinicadl.readthedocs.io/en/stable/\n"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"ClinicaDL version {__version__}",
        help="Show the ClinicaDL version and exit.",
    )

    args = parser.parse_args()

    if not vars(args):
        print("ClinicaDL is a deep learning library for neuroimaging analysis.")
        print("Use `clinicadl --help` to see available options.")
