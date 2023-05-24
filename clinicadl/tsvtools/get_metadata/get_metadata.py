from logging import getLogger
from pathlib import Path

import numpy as np
import pandas as pd

from clinicadl.utils.exceptions import ClinicaDLArgumentError, ClinicaDLTSVError
from clinicadl.utils.tsvtools_utils import merged_tsv_reader

logger = getLogger("clinicadl.tsvtools.get_metadata")


def get_metadata(
    data_tsv: Path, merged_tsv: Path, variables_of_interest=None
) -> pd.DataFrame:
    """
    Get the meta data in metadata_df to write them in output_df.
    If variables_of_interest is None, the function writes all the data that are in metadata_df for the list of subjects in output_df.

    Parameters
    ----------
    data_tsv: str (Path)
        Columns must include ['participant_id', 'session_id']
    merged_tsv: str (Path)
        Output of `clinica merge-tsv`
    variables_of_interest: list of str
        List of columns that will be added in the output DataFrame.

    Returns
    -------
    """

    metadata_df = merged_tsv_reader(merged_tsv)
    in_out_df = merged_tsv_reader(data_tsv)

    variables_in = in_out_df.columns.tolist()
    variables_metadata = metadata_df.columns.tolist()

    variables_intersection = list(
        set(variables_metadata).intersection(set(variables_in))
    )

    if variables_of_interest is None:
        variables_list = np.unique(variables_metadata)
        logger.debug(
            f"Adding the following columns to the input tsv file: {variables_list}"
        )
        result_df = pd.merge(metadata_df, in_out_df, on=variables_intersection)
        result_df.set_index(["participant_id", "session_id"], inplace=True)

    else:
        if not set(variables_of_interest).issubset(set(metadata_df.columns.values)):
            raise ClinicaDLArgumentError(
                f"The variables asked by the user {variables_of_interest} do not "
                f"exist in the data set."
            )
        else:
            variables_of_interest = list(variables_of_interest)
            variables_list = np.unique(variables_of_interest + variables_in)
            logger.debug(
                f"Adding the following columns to the input tsv file: {variables_list}"
            )
            result_df = pd.merge(metadata_df, in_out_df, on=variables_intersection)
            result_df = result_df[variables_list]
            result_df.set_index(["participant_id", "session_id"], inplace=True)

    result_df.to_csv(data_tsv, sep="\t")

    logger.info(f"metadata were added in: {data_tsv}")
