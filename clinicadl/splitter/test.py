from pathlib import Path

import pandas as pd
import torchio.transforms as transforms

from clinicadl.data.dataloader.config import DataLoaderConfig
from clinicadl.data.datasets.caps_dataset import CapsDataset
from clinicadl.data.datatype.preprocessing import T1Linear
from clinicadl.splitter import make_kfold, make_split
from clinicadl.splitter.splitter import KFold, SingleSplit
from clinicadl.transforms.extraction import Image, Patch, Slice
from clinicadl.transforms.transforms import Transforms
from clinicadl.tsvtools.get_metadata.get_metadata import get_metadata

# maps_path = Path("/")
# manager = ExperimentManager(maps_path, overwrite=False)


sub_ses_t1 = Path("/Users/camille.brianceau/aramis/CLINICADL/caps/subjects_t1.tsv")
sub_ses_all = Path("/Users/camille.brianceau/aramis/CLINICADL/caps/subjects.tsv")

# df = get_metadata(sub_ses_t1, sub_ses_all)

splir_dir = make_split(
    sub_ses_t1,
    output_dir=Path(
        "/Users/camille.brianceau/aramis/CLINICADL/clinicadl/tests/unittests/ressources/caps_example/split_test"
    ),
    subset_name="test",
    stratification=["age", "sex", "test", "diagnosis"],
    n_test=0.2,
)
print(splir_dir)


train_path = splir_dir / "train_baseline.tsv"
test_path = splir_dir / "test_baseline.tsv"

train_df = pd.read_csv(train_path, sep="\t")
test_df = pd.read_csv(test_path, sep="\t")

print("train age mean", train_df["age"].mean())
print("test age mean", test_df["age"].mean())
print("/n")
print("train age std", train_df["age"].std())
print("test age std", test_df["age"].std())
print("/n")
print("train test mean", train_df["test"].mean())
print("test test mean", test_df["test"].mean())
print("/n")
print("train test std", train_df["test"].std())
print("test test std", test_df["test"].std())

print("/n")
print(
    "train diagnosis count AD",
    len(train_df[train_df["diagnosis"] == "AD"]),
    "/",
    len(train_df),
    len(train_df[train_df["diagnosis"] == "AD"]) / len(train_df),
)
print(
    "test diagnosis count AD",
    len(test_df[test_df["diagnosis"] == "AD"]),
    "/",
    len(test_df),
    len(test_df[test_df["diagnosis"] == "AD"]) / len(test_df),
)
print("/n")
print(
    "train diagnosis count MCI",
    len(train_df[train_df["diagnosis"] == "MCI"]),
    "/",
    len(train_df),
    len(train_df[train_df["diagnosis"] == "MCI"]) / len(train_df),
)
print(
    "test diagnosis count MCI",
    len(test_df[test_df["diagnosis"] == "MCI"]),
    "/",
    len(test_df),
    len(test_df[test_df["diagnosis"] == "MCI"]) / len(test_df),
)
print("/n")
print(
    "train diagnosis count CN",
    len(train_df[train_df["diagnosis"] == "CN"]),
    "/",
    len(train_df),
    len(train_df[train_df["diagnosis"] == "CN"]) / len(train_df),
)
print(
    "test diagnosis count CN",
    len(test_df[test_df["diagnosis"] == "CN"]),
    "/",
    len(test_df),
    len(test_df[test_df["diagnosis"] == "CN"]) / len(test_df),
)

print("/n")
print(
    "train sex count F",
    len(train_df[train_df["sex"] == "F"]),
    "/",
    len(train_df),
    len(train_df[train_df["sex"] == "F"]) / len(train_df),
)
print(
    "test sex count F",
    len(test_df[test_df["sex"] == "F"]),
    "/",
    len(test_df),
    len(test_df[test_df["sex"] == "F"]) / len(test_df),
)
print("/n")
print(
    "train sex count M",
    len(train_df[train_df["sex"] == "M"]),
    "/",
    len(train_df),
    len(train_df[train_df["sex"] == "M"]) / len(train_df),
)
print(
    "test sex count M",
    len(test_df[test_df["sex"] == "M"]),
    "/",
    len(test_df),
    len(test_df[test_df["sex"] == "M"]) / len(test_df),
)


fold_dir = make_kfold(train_path, stratification="sex", n_splits=2)

print(fold_dir)

caps_directory = Path("/Users/camille.brianceau/aramis/CLINICADL/caps")
preprocessing_t1 = T1Linear()
transforms_image = Transforms(
    image_augmentation=[transforms.RandomMotion()],
    extraction=Image(),
    image_transforms=[transforms.Blur((0.5, 0.6, 0.3))],
)
dataset_t1_image = CapsDataset(
    caps_directory=caps_directory,
    data=train_path,
    preprocessing=preprocessing_t1,
    transforms=transforms_image,
)
print(dataset_t1_image.__str__())
dataset_t1_image.prepare_data(n_proc=2)

splitter = KFold(fold_dir)

for split in splitter.get_splits(dataset=dataset_t1_image):
    print(f"Split {split.index}:\n")
    print(f"Train dataset: {split.train_dataset}")
    print(f"describe: {split.train_dataset.describe()}")
    print(f"elem per image: {split.train_dataset.elem_per_image}")
    print("\n")
    print(f"Validation dataset: {split.val_dataset}")
    print(f"describe: {split.val_dataset.describe()}")
    print(f"elem per image: {split.val_dataset.elem_per_image}")

    split.build_train_loader(num_workers=2)
    split.build_val_loader(DataLoaderConfig(batch_size=2))
    print("/n")
    print(f"Train loader: {split.train_loader}")
    print(f"Validation loader: {split.val_loader}")
