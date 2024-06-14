import tarfile
from pathlib import Path
from typing import Optional, Tuple

from clinicadl.utils.clinica_utils import (
    RemoteFileStructure,
    fetch_file,
)
from clinicadl.utils.enum import MaskChecksum, Pathology
from clinicadl.utils.exceptions import ClinicaDLArgumentError, DownloadError


def get_info_from_filename(image_path: Path) -> Tuple[str, str, str, str]:
    input_filename = image_path.name
    filename = "_".join(input_filename.split("_")[2::])
    subject_name = input_filename.split("_")[:1][0]
    session_name = input_filename.split("_")[1:2][0]

    if filename.endswith(".nii.gz"):
        file_suffix = ".nii.gz"
        filename_pattern = Path(Path(filename).stem).stem
    elif filename.endswith(".nii"):
        file_suffix = ".nii"
        filename_pattern = Path(filename).stem

    return subject_name, session_name, filename_pattern, file_suffix


def get_mask_checksum_and_filename(
    pathology: Optional[Pathology],
) -> Tuple[MaskChecksum, str]:
    if pathology is None:
        filename = "AAL2.tar.gz"
        checksum = MaskChecksum.AAL2
    else:
        filename = f"mask_hypo_{pathology.value}.nii"
        if pathology == Pathology.PCA:
            checksum = MaskChecksum.PCA
        elif pathology == Pathology.AD:
            checksum = MaskChecksum.AD
        elif pathology == Pathology.BVFTD:
            checksum = MaskChecksum.BVFTD
        elif pathology == Pathology.SVPPA:
            checksum = MaskChecksum.SVPPA
        elif pathology == Pathology.NFVPPA:
            checksum = MaskChecksum.NFVPPA
        elif pathology == Pathology.LVPPA:
            checksum = MaskChecksum.LVPPA
        else:
            raise ClinicaDLArgumentError("invalid pathology for checksum")
    return checksum, filename


def get_mask_path(pathology: Optional[Pathology] = None) -> Path:
    if pathology is None:
        cache_clinicadl = Path.home() / ".cache" / "clinicadl" / "ressources" / "masks"  # noqa (typo in resources)
        url_aramis = "https://aramislab.paris.inria.fr/files/data/masks/"
    else:
        cache_clinicadl = (
            Path.home() / ".cache" / "clinicadl" / "ressources" / "masks_hypo"
        )  # noqa (typo in resources)
        url_aramis = "https://aramislab.paris.inria.fr/files/data/masks/hypo/"

    checksum, filename = get_mask_checksum_and_filename(pathology)
    FILE1 = RemoteFileStructure(
        filename=filename,
        url=url_aramis,
        checksum=checksum.value,
    )
    cache_clinicadl.mkdir(parents=True, exist_ok=True)
    if not ((cache_clinicadl / "AAL2").is_dir()) and not (
        (cache_clinicadl / filename).is_file()
    ):
        print("Downloading masks...")
        try:
            mask_path_tar = fetch_file(FILE1, cache_clinicadl)
            if pathology is None:
                tar_file = tarfile.open(mask_path_tar)
                print(f"File: {mask_path_tar}")
                try:
                    tar_file.extractall(cache_clinicadl)
                    tar_file.close()
                    v = cache_clinicadl / "AAL2"
                except RuntimeError:
                    print("Unable to extract downloaded files.")
            else:
                v = mask_path_tar
        except IOError as err:
            print("Unable to download required templates:", err)
            raise DownloadError(
                """Unable to download masks, please download them
                manually at https://aramislab.paris.inria.fr/files/data/masks/
                and provide a valid path."""
            )
    else:
        if pathology is None:
            v = cache_clinicadl / "AAL2"
        else:
            v = cache_clinicadl / filename
    return v
