from clinicadl.data.structures import DataPoint, Sample, Sample2D
from clinicadl.data.structures.examples import (
    ColinDataPoint,
    ColinSample,
    ColinSample2D,
)


def test_ColinDataPoint():
    colin = ColinDataPoint()
    assert isinstance(colin, DataPoint)
    assert colin.shape == (1, 181, 217, 181)
    colin = ColinDataPoint(participant="sub-abc")
    assert colin.participant == "sub-abc"


def test_ColinSample():
    colin = ColinSample()
    assert isinstance(colin, Sample)
    assert colin.shape == (1, 181, 217, 181)
    colin = ColinSample(participant="sub-abc")
    assert colin.participant == "sub-abc"


def test_ColinSample2D():
    colin = ColinSample2D()
    assert isinstance(colin, Sample2D)
    assert colin.shape == (1, 181, 1, 181)
    colin = ColinSample2D(participant="sub-abc")
    assert colin.participant == "sub-abc"
