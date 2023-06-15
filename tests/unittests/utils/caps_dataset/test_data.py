import re

import pandas as pd
import pytest


@pytest.mark.parametrize(
    "dataframe_columns,purpose",
    [
        (["foo", "cohort", "bar", "baz", "path"], "caps"),
        (["diagnoses", "foo", "cohort", "bar", "baz", "path"], "caps"),
        (["diagnoses", "foo", "cohort", "bar", "baz", "path"], "foo"),
    ],
)
def test_check_multi_cohort_tsv(dataframe_columns, purpose):
    from clinicadl.utils.caps_dataset.data import check_multi_cohort_tsv

    assert (
        check_multi_cohort_tsv(pd.DataFrame(columns=dataframe_columns), purpose) is None
    )


@pytest.mark.parametrize(
    "dataframe_columns,purpose,expected_mandatory_columns",
    [
        (
            ["foo", "cohort", "bar", "baz", "path"],
            "bids",
            ("cohort", "diagnoses", "path"),
        ),
        (
            ["foo", "baz", "path"],
            "caps",
            ("cohort", "path"),
        ),
    ],
)
def test_check_multi_cohort_tsv_errors(
    dataframe_columns, purpose, expected_mandatory_columns
):
    from clinicadl.utils.caps_dataset.data import check_multi_cohort_tsv
    from clinicadl.utils.exceptions import ClinicaDLTSVError

    with pytest.raises(
        ClinicaDLTSVError,
        match=re.escape(
            f"Columns of the TSV file used for {purpose} location "
            f"must include {expected_mandatory_columns}"
        ),
    ):
        check_multi_cohort_tsv(pd.DataFrame(columns=dataframe_columns), purpose)
