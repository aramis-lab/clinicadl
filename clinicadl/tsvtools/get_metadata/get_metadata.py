from logging import getLogger

import numpy as np
import pandas as pd

from clinicadl.utils.tsvtools_utils import merge_tsv_reader

logger = getLogger("clinicadl")


def get_metadata(
    metadata_df: DataFrame, output_df: DataFrame, variables_of_interest=None
) -> DataFrame:
    """
    Get the meta data in metadata_df to write them in output_df.
    If variables_of_interest is None, the function writes all the data that are in metadata_df for the list of subjects in output_df.

    Args:
        data_df: DataFrame with columns including ['participant_id', 'session_id']
        variables_of_interest : (str) list of columns that should be kept in the output DataFrame.

    Returns:
        output_df : data_df DataFrame with variables of interest columns added.
    """

    variables_in = output_df.columns.tolist()
    variables_metadata = metadata_df.columns.tolist()

    variables_intersection = list(
        set(variables_metadata).intersection(set(variables_in))
    )

    if variables_of_interest is None:

        variables_list = np.unique(variables_in)
        result_df = pd.merge(metadata_df, output_df, on=variables_intersection)
        result_df.set_index(["participant_id", "session_id"], inplace=True)
        result_df.to_csv(output_tsv, sep="\t")

    else:

        if not set(variables_of_interest).issubset(set(metadata_df.columns.values)):
            raise ClinicaDLArgumentError(
                f"The variables asked by the user {variables_of_interest} do not "
                f"exist in the data set."
            )
        else:
            variables_of_interest = list(variables_of_interest)
            variables_list = np.unique(variables_of_interest + variables_in)
            result_df = pd.merge(metadata_df, output_df, on=variables_intersection)
            result_df = result_df[variables_list]
            result_df.set_index(["participant_id", "session_id"], inplace=True)
    return result_df


# input : tsv file wanted
# output : tsv file with more metadata
