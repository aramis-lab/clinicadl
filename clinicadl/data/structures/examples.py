from pathlib import Path

import torchio as tio
from torchio.datasets import Colin27

from .datapoint import DataPoint
from .sample import Sample, Sample2D


class Colin27DataPoint(DataPoint):
    """
    Example of a :py:class:`~clinicadl.data.structures.DataPoint`.

    It contains a T1 image and a mask called "head".

    The default fields can be overwritten and additional fields can be added.

    Examples
    --------
    >>> from clinicadl.data.structures.examples import Colin27DataPoint
    >>> colin = Colin27DataPoint()
    >>> colin
    Colin27DataPoint(Keys: ('head', 'image', 'participant', 'session'); images: 2)
    >>> colin.participant
    'sub-colin'
    >>> colin = Colin27DataPoint(participant="sub-000", new_field="x")
    >>> colin.participant
    'sub-000'
    >>> colin["new_field"]
    'x'
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


class Colin27Sample(Sample):
    """
    Example of a :py:class:`~clinicadl.data.structures.Sample`.

    It contains a T1 image and an mask called "head".

    The default fields can be overwritten and additional fields can be added.

    Examples
    --------
    >>> from clinicadl.data.structures.examples import Colin27Sample
    >>> colin = Colin27Sample()
    >>> colin
    Colin27Sample(Keys: ('head', 'file_type', 'image_path', 'sample_type', 'sample_position', 'image', 'participant', 'session'); images: 2)
    >>> colin.participant
    'sub-colin'
    >>> colin = Colin27Sample(participant="sub-000", new_field="x")
    >>> colin.participant
    'sub-000'
    >>> colin["new_field"]
    'x'
    """

    def __init__(self, **kwargs):
        from clinicadl.io.bids import BidsFileType

        tio_colin = Colin27()
        # pylint: disable=no-member
        args = {
            "image": tio_colin.t1,
            "head": tio_colin.head,
            "participant": "sub-colin",
            "session": "ses-M000",
            "file_type": BidsFileType(data_type="t1", suffix="T1w"),
            "image_path": Path("bids")
            / "sub-000"
            / "ses-M000"
            / "anat"
            / "sub-000_ses-M000_T1w.nii.gz",
        }
        args.update(kwargs)
        # pylint: disable=no-member
        super().__init__(**args)


class Colin27Sample2D(Sample2D):
    """
    Example of a :py:class:`~clinicadl.data.structures.Sample2D`.

    It contains a T1 image and an additional mask called "head".

    The default fields can be overwritten and additional fields can be added.

    Examples
    --------
    >>> from clinicadl.data.structures.examples import Colin27Sample2D
    >>> colin = Colin27Sample2D()
    >>> colin
    Colin27Sample2D(Keys: ('head', 'slice_direction', 'squeeze', 'file_type', 'image_path', 'sample_type', 'sample_position', 'image', 'participant', 'session'); images: 2)
    >>> colin.participant
    'sub-colin'
    >>> colin = Colin27Sample2D(participant="sub-000", new_field="x")
    >>> colin.participant
    'sub-000'
    >>> colin["new_field"]
    'x'
    """

    def __init__(self, **kwargs):
        from clinicadl.io.bids import BidsFileType

        tio_colin = Colin27()
        tio_colin = tio.CropOrPad(target_shape=(181, 1, 181))(tio_colin)
        # pylint: disable=no-member
        args = {
            "image": tio_colin.t1,
            "head": tio_colin.head,
            "participant": "sub-colin",
            "session": "ses-M000",
            "file_type": BidsFileType(data_type="t1", suffix="T1w"),
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
