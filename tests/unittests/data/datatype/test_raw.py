from clinicadl.data.datatype.raw import RawCustom, RawDWI, RawFlair, RawPET, RawT1w


def test_good_custom():
    custom_data = RawCustom(custom_suffix="example")
    assert custom_data.custom_suffix == "example"
    assert custom_data.modality == "custom"
    assert custom_data.file_type.pattern == "sub-*_ses-*_example.nii*"
    assert (
        custom_data.file_type.description
        == "Raw custom NIfTI images with suffix 'example'"
    )
    assert custom_data.file_type.needed_pipeline is None
    assert str(custom_data) == "Raw custom NIfTI images with suffix 'example'"


def test_good_flair():
    flair_data = RawFlair()
    assert flair_data.modality == "flair"
    assert flair_data.file_type.pattern == "anat/sub-*_ses-*_flair.nii*"
    assert flair_data.file_type.description == "Raw FLAIR T2w MRI NIfTI images"
    assert flair_data.file_type.needed_pipeline is None


def test_good_t1():
    t1w_data = RawT1w()
    assert t1w_data.modality == "T1w"
    assert t1w_data.file_type.pattern == "anat/sub-*_ses-*_T1w.nii*"
    assert t1w_data.file_type.description == "Raw T1w MRI NIfTI images"
    assert t1w_data.file_type.needed_pipeline is None


def test_good_pet():
    pet_data = RawPET(tracer="18FAV45", reconstruction="nacstat")
    assert pet_data.modality == "pet"
    assert pet_data.tracer == "18FAV45"
    assert pet_data.reconstruction == "nacstat"
    assert (
        pet_data.file_type.pattern == "pet/sub-*_ses-*_trc-18FAV45_rec-nacstat_pet.nii*"
    )
    assert (
        pet_data.file_type.description
        == "Raw PET NIfTI images with tracer '18FAV45' and reconstruction method 'nacstat'"
    )
    assert pet_data.file_type.needed_pipeline is None

    pet_data.reconstruction = None
    assert (
        pet_data.file_type.description == "Raw PET NIfTI images with tracer '18FAV45'"
    )
    assert pet_data.file_type.pattern == "pet/sub-*_ses-*_trc-18FAV45_pet.nii*"


def test_good_dwi():
    dwi_data = RawDWI()
    assert dwi_data.modality == "dwi"
    assert dwi_data.file_type.pattern == "dwi/sub-*_ses-*_dwi.nii*"
    assert dwi_data.file_type.description == "Raw DW MRI NIfTI images"
    assert dwi_data.file_type.needed_pipeline is None
