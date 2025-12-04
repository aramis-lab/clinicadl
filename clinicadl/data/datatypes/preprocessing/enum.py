from enum import Enum


class SupportedPreprocessing(str, Enum):
    """Preprocessing methods supported in ``ClinicaDL``."""

    T1_LINEAR = "T1Linear"
    PET_LINEAR = "PETLinear"
    FLAIR_LINEAR = "FlairLinear"
    DWI_DTI = "DWIDTI"

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not supported. Supported preprocessings are: "
            + ", ".join([repr(m.value) for m in cls])
            + ". Use directly clinicadl.data.datatypes.DataType to create your own datatype."
        )
