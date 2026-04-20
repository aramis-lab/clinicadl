import re

import pytest

from clinicadl.utils.bids import BidsEntity, Session, Subject


def test_bids_entity():
    with pytest.raises(
        ValueError,
        match="A BIDS entity must be of the form '<key>-<value>'. Got '18FFDG'",
    ):
        BidsEntity("18FFDG")
    with pytest.raises(
        AssertionError,
        match="They key of a BIDS entity must be an alphanumeric string. Got: 'radio_trc'",
    ):
        BidsEntity("radio_trc-18FFDG")
    tracer = BidsEntity("trc-18FFDG")
    assert tracer.key == "trc"
    assert tracer.value == "18FFDG"

    tracer = BidsEntity.from_key_value(key="trc", value=0)
    assert tracer.key == "trc"
    assert tracer.value == "0"


def test_subject():
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "A participant id must start with 'sub' (e.g., 'sub-001'). Got 'Sub'"
        ),
    ):
        Subject("Sub-001")
    sub = Subject("sub-001")
    assert sub.key == "sub"
    assert sub.value == "001"

    sub = Subject.from_value("001")
    assert sub.key == "sub"
    assert sub.value == "001"


def test_session():
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "A session id must start with 'ses' (e.g., 'ses-M000'). Got 'Ses'"
        ),
    ):
        Session("Ses-M000")
    sub = Session("ses-M000")
    assert sub.key == "ses"
    assert sub.value == "M000"

    ses = Session.from_value("M000")
    assert ses.key == "ses"
    assert ses.value == "M000"
