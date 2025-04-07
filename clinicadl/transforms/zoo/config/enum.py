from clinicadl.utils.enum import BaseEnum


class ZooTransform(str, BaseEnum):
    """
    Implemented transforms in ClinicaDL Transform Zoo.
    see: https://torchio.readthedocs.io/transforms/transforms.html
    """

    NAN_REMOVAL = "NanRemoval"
