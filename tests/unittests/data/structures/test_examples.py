import torchio as tio

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
    transformed = tio.RescaleIntensity()(colin)
    assert transformed.image.tensor.max().item() == 1
    assert colin.image.tensor.max().item() != 1
