import shutil
from pathlib import Path

import pandas as pd

from clinicadl.utils.exceptions import ClinicaDLArgumentError


def mood_to_clinicadl(mood_path: Path, caps_path: Path, caps: bool = False):
    if not isinstance(caps_path, Path):
        caps_path = Path(caps_path)

    if not isinstance(mood_path, Path):
        mood_path = Path(mood_path)

    if mood_path.is_dir():
        path_list = list(mood_path.glob("*.nii.gz"))
    else:
        raise ClinicaDLArgumentError(
            f"The path you give is not a directory: {mood_path}"
        )

    if caps_path.is_dir():
        raise ClinicaDLArgumentError(f"The path you give already exists: {caps_path}")
    else:
        (caps_path).mkdir(parents=True)

    columns = ["participant_id", "session_id", "diagnosis"]
    output_df = pd.DataFrame(columns=columns)
    session = "ses-M000"
    diagnosis = "CN"
    for mood_path in path_list:
        print(Path(mood_path.stem).stem.split("_"))
        subject = int(Path(mood_path.stem).stem)
        if subject < 10:
            subject_num = "00" + str(subject)
        elif 10 <= subject < 100:
            subject_num = "0" + str(subject)
        elif 100 <= subject:
            subject_num = str(subject)

        subject = "sub-" + subject_num

        if caps:
            subject_path = caps_path / "subjects" / subject / session / "custom"
            filename = f"sub-{subject_num}_ses-M000_mood.nii.gz"

        else:
            subject_path = caps_path / subject / session / "custom"
            filename = f"sub-{subject_num}_ses-M000_mood.nii.gz"

        row_df = pd.DataFrame([[subject, session, diagnosis]], columns=columns)
        output_df = pd.concat([output_df, row_df])

        subject_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src=mood_path, dst=subject_path / filename)

    output_df.to_csv(caps_path / "subjects_sessions.tsv", sep="\t", index=False)

    return caps_path / "subjects_sessions.tsv"
