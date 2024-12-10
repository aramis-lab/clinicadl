from pathlib import Path

import torchio.transforms as transforms

from clinicadl.dataset.datasets.caps_dataset import CapsDataset
from clinicadl.dataset.preprocessing import PreprocessingT1
from clinicadl.splitter import make_kfold, make_split
from clinicadl.splitter.dataloader import DataLoaderConfig
from clinicadl.splitter.splitter import KFold, SingleSplit
from clinicadl.transforms.extraction import ROI, Image, Patch, Slice
from clinicadl.transforms.transforms import Transforms

# maps_path = Path("/")
# manager = ExperimentManager(maps_path, overwrite=False)


sub_ses_t1 = Path("/Users/camille.brianceau/aramis/CLINICADL/caps/subjects_t1.tsv")

splir_dir = make_split(sub_ses_t1, ignore_demographics=True, n_test=2)

print(splir_dir)

train_path = splir_dir / "train.tsv"

fold_dir = make_kfold(train_path, ignore_demographics=True, n_splits=2)

print(fold_dir)

caps_directory = Path("/Users/camille.brianceau/aramis/CLINICADL/caps")
preprocessing_t1 = PreprocessingT1()
transforms_image = Transforms(
    image_augmentation=[transforms.RandomMotion()],
    extraction=Image(),
    image_transforms=[transforms.Blur((0.5, 0.6, 0.3))],
)
dataset_t1_image = CapsDataset(
    caps_directory=caps_directory,
    data=sub_ses_t1,
    preprocessing=preprocessing_t1,
    transforms=transforms_image,
)
print(dataset_t1_image)
dataset_t1_image.prepare_data(n_proc=2)

splitter = KFold(fold_dir)

for split in splitter.get_splits(dataset=dataset_t1_image):
    print(f"Split {split.index}:")
    print(f"Train dataset: {split.train_dataset}")
    print(f"Validation dataset: {split.val_dataset}")

    split.build_train_loader(num_workers=2)
    split.build_val_loader(DataLoaderConfig(batch_size=2))

    print(f"Train loader: {split.train_loader}")
    print(f"Validation loader: {split.val_loader}")
