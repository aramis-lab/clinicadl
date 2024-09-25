from logging import getLogger
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict

from clinicadl.utils.exceptions import ClinicaDLArgumentError  # type: ignore

logger = getLogger("clinicadl.predict_config")


class MapsManagerConfig(BaseModel):
    maps_dir: Path
    data_group: Optional[str] = None
    overwrite: bool = False
    save_nifti: bool = False

    # pydantic config
    model_config = ConfigDict(validate_assignment=True)

    def check_output_saving_nifti(self, network_task: str) -> None:
        # Check if task is reconstruction for "save_tensor" and "save_nifti"
        if self.save_nifti and network_task != "reconstruction":
            raise ClinicaDLArgumentError(
                "Cannot save nifti if the network task is not reconstruction. Please remove --save_nifti option."
            )


def init_split_manager(
    validation,
    parameters,
    split_list=None,
    ssda_bool: bool = False,
    caps_target: Optional[Path] = None,
    tsv_target_lab: Optional[Path] = None,
):
    from clinicadl.validation import split_manager

    split_class = getattr(split_manager, validation)
    args = list(
        split_class.__init__.__code__.co_varnames[
            : split_class.__init__.__code__.co_argcount
        ]
    )
    args.remove("self")
    args.remove("split_list")
    kwargs = {"split_list": split_list}
    for arg in args:
        kwargs[arg] = parameters[arg]

    if ssda_bool:
        kwargs["caps_directory"] = caps_target
        kwargs["tsv_path"] = tsv_target_lab

    return split_class(**kwargs)
