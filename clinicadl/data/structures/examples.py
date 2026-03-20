from pathlib import Path

import torchio as tio
from torchio.datasets import Colin27

from .datapoint import DataPoint
from .sample import Sample, Sample2D


class ColinDataPoint(DataPoint):
    """
    Example of a :py:class:`~clinicadl.data.structures.DataPoint`.

    It contains a T1 image and a mask called "head".

    The default fields can be overwritten.

    Examples
    --------
    >>> from clinicadl.data.structures.examples import ColinDataPoint
    >>> colin = ColinDataPoint()
    >>> colin
    ColinDataPoint(Keys: ('head', 'image', 'participant', 'session'); images: 2)
    >>> colin.participant
    'sub-colin'
    >>> colin = ColinDataPoint(participant="sub-000")
    >>> colin.participant
    'sub-000'
    """

    def __init__(self, **kwargs):
        tio_colin = Colin27()
        # pylint: disable=no-member
        args = {
            "image": tio_colin.t1,
            "head": tio_colin.head,
            "participant": "sub-colin",
            "session": "ses-M000",
        }
        args.update(kwargs)
        super().__init__(**args)


class ColinSample(Sample):
    """
    Example of a :py:class:`~clinicadl.data.structures.Sample`.

    It contains a T1 image and an mask called "head".

    The default fields can be overwritten.

    Examples
    --------
    >>> from clinicadl.data.structures.examples import ColinSample
    >>> colin = ColinSample()
    >>> colin
    ColinSample(Keys: ('head', 'datatype', 'image_path', 'sample_type', 'sample_position', 'image', 'participant', 'session'); images: 2)
    >>> colin.participant
    'sub-colin'
    >>> colin = ColinSample(participant="sub-000")
    >>> colin.participant
    'sub-000'
    """

    def __init__(self, **kwargs):
        from clinicadl.data.datatypes import DataType

        tio_colin = Colin27()
        # pylint: disable=no-member
        args = {
            "image": tio_colin.t1,
            "head": tio_colin.head,
            "participant": "sub-colin",
            "session": "ses-M000",
            "datatype": DataType.from_folder_and_suffix(folder="t1", suffix="T1w"),
            "image_path": Path("bids")
            / "sub-000"
            / "ses-M000"
            / "anat"
            / "sub-000_ses-M000_T1w.nii.gz",
        }
        args.update(kwargs)
        # pylint: disable=no-member
        super().__init__(**args)


class ColinSample2D(Sample2D):
    """
    Example of a :py:class:`~clinicadl.data.structures.Sample2D`.

    It contains a T1 image and an additional mask called "head".

    The default fields can be overwritten.

    Examples
    --------
    >>> from clinicadl.data.structures.examples import ColinSample2D
    >>> colin = ColinSample2D()
    >>> colin
    ColinSample2D(Keys: ('head', 'slice_direction', 'squeeze', 'datatype', 'image_path', 'sample_type', 'sample_position', 'image', 'participant', 'session'); images: 2)
    >>> colin.participant
    'sub-colin'
    >>> colin = ColinSample2D(participant="sub-000")
    >>> colin.participant
    'sub-000'
    """

    def __init__(self, **kwargs):
        from clinicadl.data.datatypes import DataType

        tio_colin = Colin27()
        tio_colin = tio.CropOrPad(target_shape=(181, 1, 181))(tio_colin)
        # pylint: disable=no-member
        args = {
            "image": tio_colin.t1,
            "head": tio_colin.head,
            "participant": "sub-colin",
            "session": "ses-M000",
            "datatype": DataType.from_folder_and_suffix(folder="t1", suffix="T1w"),
            "image_path": Path("bids")
            / "sub-000"
            / "ses-M000"
            / "anat"
            / "sub-000_ses-M000_T1w.nii.gz",
            "sample_position": 108,
            "slice_direction": 1,
            "squeeze": True,
        }
        args.update(kwargs)
        # pylint: disable=no-member
        super().__init__(**args)
