from typing import Optional, Sequence, Union

from pydantic import Field, NonNegativeInt

from clinicadl.utils.dictionary.words import NAME_
from clinicadl.utils.dtype import DtypeLike, read_dtype
from clinicadl.utils.factories import get_defaults_from

from ..homemade import Format, MergeFields
from .base import TransformConfig

__all__ = [
    "FormatConfig",
    "MergeFieldsConfig",
]

FORMAT_DEFAULTS = get_defaults_from(Format)


class FormatConfig(TransformConfig):
    """
    Config class for :py:class:`clinicadl.transforms.Format`.
    """

    dtype: Optional[DtypeLike] = Field(
        default=FORMAT_DEFAULTS["dtype"], reader=read_dtype
    )
    squeeze: Union[bool, NonNegativeInt, Sequence[NonNegativeInt]] = FORMAT_DEFAULTS[
        "squeeze"
    ]
    unsqueeze: Optional[NonNegativeInt] = FORMAT_DEFAULTS["unsqueeze"]

    @classmethod
    def _get_class(cls):
        return Format


class MergeFieldsConfig(TransformConfig):
    """
    Config class for :py:class:`clinicadl.transforms.MergeFields`.
    """

    keys: Sequence[str]
    output_key: str

    @classmethod
    def _get_class(cls):
        return MergeFields

    def get_object(self, **kwargs):
        return MergeFields(*self.keys, **self.to_dict(exclude=[NAME_, "keys"]))
