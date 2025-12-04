from enum import Enum


class ImageModality(str, Enum):
    """Image modalities supported natively in ``ClinicaDL``."""

    T1W = "T1w"
    PET = "pet"
    FLAIR = "FLAIR"
