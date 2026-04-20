import pandas as pd

from clinicadl.utils.tsvtools import create_participants_sessions_df


def test_create_participants_sessions_df():
    df = create_participants_sessions_df(
        [("sub-000", "ses-M000"), ("sub-100", "ses-M003")]
    )
    pd.testing.assert_frame_equal(
        df,
        pd.DataFrame(
            {
                "participant_id": ["sub-000", "sub-100"],
                "session_id": ["ses-M000", "ses-M003"],
            }
        ),
    )
