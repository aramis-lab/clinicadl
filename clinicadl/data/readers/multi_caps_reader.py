from pathlib import Path

import pandas as pd

from clinicadl.utils.exceptions import ClinicaDLTSVError

from .caps_reader import CapsReader


class CapsMultiReader:
    def __init__(
        self,
        caps_tsv: Path,
    ):
        """CAPS reader for handling multi-cohort CAPS directories."""

        self.caps_dict = self.create_caps_dict(caps_tsv)

    def create_caps_dict(self, caps_tsv: Path) -> dict:
        if not caps_tsv.is_file():
            raise FileNotFoundError(
                f"The provided caps directory {caps_tsv} does not exist. Careful: It must be a tsv file in multi-cohort."
            )

        caps_df = pd.read_csv(caps_tsv, sep="\t")
        if not set(("cohort", "path")).issubset(caps_df.columns.values):
            raise ClinicaDLTSVError(
                "Columns of the TSV file used for CAPS location must include cohort and path"
            )

        caps_dict = {}
        for idx in range(len(caps_df)):
            cohort_name = caps_df.at[idx, "cohort"]
            cohort_path = Path(caps_df.at[idx, "path"])

            caps_reader = CapsReader(caps_directory=cohort_path)
            caps_dict[cohort_name] = caps_reader

        return caps_dict
