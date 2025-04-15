from clinicadl.data.structures import DataPoint
from clinicadl.data.structures.examples import ColinDataPoint


def test_ColinDataPoint():
    colin = ColinDataPoint()
    assert isinstance(colin, DataPoint)
    assert colin.image.shape == (1, 181, 217, 181)
    assert colin.label.shape == (1, 181, 217, 181)
    assert colin.head.shape == (1, 181, 217, 181)
    assert colin.participant == "sub-colin"
    assert colin.session == "ses-M000"
