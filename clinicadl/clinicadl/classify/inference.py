from ..tools.deep_learning import create_model, load_model, read_json
from ..tools.deep_learning.data import MRIDataset, MinMaxNormalization
from ..subject_level.utils import test
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader


def inference_from_model(caps_dir,
                         tsv_file,
                         output_dir,
                         model_path,
                         json_path,
                         options):
    """
    Inference from trained model

    This functions uses a previously trained model to classify the input

    Args:

    caps_dir: folder containing the tensor files (.pt version of MRI)
    tsv_file: file with the name of the MRIs to process (single or multiple)
    working_dir: folder containing temporary files
    model_path: name of the model to use for classification
    json_path: path to the json file describing the model
    options: Namespace for the argparse object

    Returns:

    pd frame with the prediction results

    Rises:


    """

    print("This is the inference phase")
    options = read_json(options, "CNN", json_path=json_path)

    options.caps_dir = caps.dir
    # Recreate the model with the network described in the json file
    model = create_model(options.network)
    criterion = nn.CrossEntropyLoss()

    # Load model from path
    best_model, best_epoch = load_model(
            model, model_dir,
            options.gpu, filename='model_best.pth.tar')

    if options.minmaxnormalization:
        transformations = MinMaxNormalization()
    else:
        transformations = None

    data_to_test = MRIDataset(
            options.caps_dir,
            options.tsv_file,
            options.preprocessing,
            transform=transformations)

    test_loader = DataLoader(
            data_to_test,
            batch_size=options.batch_size,
            shuffle=False,
            num_workers=options.num_workers,
            pin_memory=True)

    metrics_test, loss_test, test_df = test(
            best_model,
            test_loader,
            options.gpu,
            criterion,
            full_return=True)

    return testdf
