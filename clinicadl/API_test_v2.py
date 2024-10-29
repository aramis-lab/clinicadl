from pathlib import Path
from clinicadl.caps_dataset2.config.preprocessing import PreprocessingConfig
from clinicadl.caps_dataset2.config.extraction import ExtractionConfig
from clinicadl.caps_dataset2.data import CapsDatasetRoi, CapsDatasetPatch, CapsDatasetSlice
from clinicadl.transforms.config import TransformsConfig
import torchio
from clinicadl import tsvtools
from clinicadl.trainer.trainer import Trainer

class ExperimentManager:
    pass

class CapsReader:
    pass

class Transforms:
    pass

class Predictor:
    pass

class ClinicaDLModel:
    pass

class KFolder:
    pass

def get_loss_function():
    pass

def get_network_from_config():
    pass

def create_network_config():
    pass

def get_single_split():
    pass

# Create the Maps Manager / Read/write manager /
maps_path = Path("/")
manager = ExperimentManager(maps_path, overwrite = False)

caps_directory = Path("caps_directory") # output of clinica pipelines
caps_reader = CapsReader(caps_directory, manager= manager)

preprocessing_1: PreprocessingConfig = caps_reader.get_preprocessing("t1-linear")
extraction_1: ExtractionConfig = caps_reader.extract_slice(preprocessing=preprocessing_1, arg_slice = 2)
transforms_1 = Transforms(data_augmentation = [torchio.t1, torchio.t2], image_transforms= [torchio.t1, torchio.t2], object_transforms=[torchio.t1, torchio.t2]) # not mandatory 

preprocessing_2: PreprocessingConfig = caps_reader.get_preprocessing("pet-linear")
extraction_2: ExtractionConfig= caps_reader.extract_patch(preprocessing=preprocessing_2, arg_patch = 2)
transforms_2 = Transforms(data_augmentation = [torchio.t2], image_transforms= [torchio.t1], object_transforms=[torchio.t1, torchio.t2]) 

sub_ses_tsv = Path("")
split_dir = tsvtools.split_tsv(sub_ses_tsv) #-> creer un test.tsv et un train.tsv

dataset_t1_roi: CapsDatasetRoi  = caps_reader.get_dataset( extraction = extraction_1, preprocessing = preprocessing_1, sub_ses_tsv = split_dir / "train.tsv", transforms = transforms_1)
dataset_pet_patch: CapsDatasetPatch = caps_reader.get_dataset(extraction = extraction_2, preprocessing= preprocessing_2, sub_ses_tsv = split_dir / "train.tsv", transforms = transforms_2)

dataset_multi_modality_multi_extract = concat_dataset(dataset_t1, dataset_pet) # 2 train.tsv en entrée qu'il faut concat et pareil pour les transforms à faire attention

config_file = Path("config_file")
trainer = Trainer.from_json(config_file=config_file, manager=manager)

# CAS CROSS-VALIDATION
splitter = KFolder(n_splits = 3, caps_dataset=dataset_multi_modality_multi_extract, manager = manager)

for split in splitter.split_iterator(split_list = [0,1]):
    # bien définir ce qu'il y a dans l'objet split

    loss, loss_config = get_loss_function(CrossEntropyLossConfig())
    network_config = create_network_config(ImplementedNetworks.CNN)(in_shape = [2,2,2], num_outputs = 1, conv_args = ConvEncoderOptions(channels = [3, 2, 2]))
    network = get_network_from_config(network_config)
    
    model = ClinicaDLModel(
        network= network,
        loss=loss,
        optimizer= AdamConfig()
    )

    trainer.train(model, split)
    # le trainer va instancier un predictor/valdiator dans le train ou dans le init


# CAS SINGLE SPLIT
split = get_single_split(n_subject_validation = 0, caps_dataset=dataset_multi_modality_multi_extract, manager = manager)

loss, loss_config = get_loss_function(CrossEntropyLossConfig())
network_config = create_network_config(ImplementedNetworks.CNN)(in_shape = [2,2,2], num_outputs = 1, conv_args = ConvEncoderOptions(channels = [3, 2, 2]))
network = get_network_from_config(network_config)
    
model = ClinicaDLModel(
    network= network,
    loss=loss,
    optimizer= AdamConfig()
)

trainer.train(model, split)
# le trainer va instancier un predictor/valdiator dans le train ou dans le init


# TEST 

preprocessing_test: PreprocessingConfig = caps_reader.get_preprocessing("pet-linear")
extraction_test: ExtractionConfig= caps_reader.extract_patch(preprocessing=preprocessing_2, arg_patch = 2)
transforms_test = Transforms(data_augmentation = [torchio.t2], image_transforms= [torchio.t1], object_transforms=[torchio.t1, torchio.t2]) 

dataset_test: CapsDatasetROI  = caps_reader.get_dataset( extraction = extraction_test, preprocessing = preprocessing_test, sub_ses_tsv = split_dir / "test.tsv", transforms = transforms_test)

predictor = Predictor(manager= manager)
predictor.predict(dataset_test= dataset_test, split = 2)
