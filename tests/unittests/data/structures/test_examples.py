from clinicadl.data.structures import DataPoint, Sample, Sample2D
from clinicadl.data.structures.examples import (
    Colin27DataPoint,
    Colin27Sample,
    Colin27Sample2D,
)


def test_ColinDataPoint():
    colin = Colin27DataPoint()
    assert isinstance(colin, DataPoint)
    assert colin.shape == (1, 181, 217, 181)
    colin = Colin27DataPoint(participant="sub-abc")
    assert colin.participant == "sub-abc"


def test_ColinSample():
    colin = Colin27Sample()
    assert isinstance(colin, Sample)
    assert colin.shape == (1, 181, 217, 181)
    colin = Colin27Sample(participant="sub-abc")
    assert colin.participant == "sub-abc"


def test_ColinSample2D():
    colin = Colin27Sample2D()
    assert isinstance(colin, Sample2D)
    assert colin.shape == (1, 181, 1, 181)
    colin = Colin27Sample2D(participant="sub-abc")
    assert colin.participant == "sub-abc"
