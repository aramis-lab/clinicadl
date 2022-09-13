from logging import getLogger

import numpy as np
import pandas as pd

from clinicadl.utils.tsvtools_utils import merge_tsv_reader

logger = getLogger("clinicadl")


def get_metadata(formatted_data_tsv, output_tsv, variables_of_interest=None):

    metadata_df = merge_tsv_reader(formatted_data_tsv)
    output_df = merge_tsv_reader(output_tsv)

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
            result_df.to_csv(output_tsv, sep="\t")


# input : tsv file wanted
# output : tsv file with more metadata
