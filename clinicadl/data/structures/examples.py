from torchio.datasets import Colin27

from .datapoint import DataPoint


class ColinDataPoint(DataPoint):
    """
    Example of a :py:class:`~clinicadl.data.structures.DataPoint`.

    It contains a T1 image, a mask as a label, and an additional mask called "head".

    Examples
    --------
    >>> from clinicadl.data.structures.examples import ColinDataPoint
    >>> ColinDataPoint()
    ColinDataPoint(Keys: ('image', 'label', 'participant', 'session', 'head'); images: 3)
    """

    def __init__(self):
        tio_colin = Colin27()
        # pylint: disable=no-member
        super().__init__(
            image=tio_colin.t1,
            label=tio_colin.brain,
            head=tio_colin.head,
            participant="sub-colin",
            session="ses-M000",
        )
