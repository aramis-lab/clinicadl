import pytest
import os
import shutil


# Everything is tested on roi except for cnn --> multicnn (patch) as multicnn is not implemented for roi.
@pytest.fixture(
    params=[
        "transfer_smallAE_largeAE",
        "transfer_ae_ae",
        "transfer_smallAE_largeCNN",
        "transfer_ae_cnn",
        "transfer_cnn_cnn",
        "transfer_cnn_multicnn",
    ]
)
def cli_commands(request):

    if request.param == "transfer_smallAE_largeAE":
        source_task = [
            "train",
            "roi",
            "autoencoder",
            "data/dataset/random_example",
            "t1-linear",
            "data/labels_list",
            "results_source",
            "Conv5_FC3",
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--split",
            "0",
        ]
        target_task = [
            "train",
            "image",
            "autoencoder",
            "data/dataset/random_example",
            "t1-linear",
            "data/labels_list",
            "results_target",
            "Conv6_FC3",
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--split",
            "0",
            "--transfer_learning_path",
            "results_source",
        ]
    elif request.param == "transfer_ae_ae":
        source_task = [
            "train",
            "roi",
            "autoencoder",
            "data/dataset/random_example",
            "t1-linear",
            "data/labels_list",
            "results_source",
            "Conv4_FC3",
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--split",
            "0",
        ]
        target_task = [
            "train",
            "roi",
            "autoencoder",
            "data/dataset/random_example",
            "t1-linear",
            "data/labels_list",
            "results_target",
            "Conv4_FC3",
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--split",
            "0",
            "--transfer_learning_path",
            "results_source",
        ]
    elif request.param == "transfer_smallAE_largeCNN":
        source_task = [
            "train",
            "roi",
            "autoencoder",
            "data/dataset/random_example",
            "t1-linear",
            "data/labels_list",
            "results_source",
            "Conv5_FC3",
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--split",
            "0",
        ]
        target_task = [
            "train",
            "image",
            "cnn",
            "data/dataset/random_example",
            "t1-linear",
            "data/labels_list",
            "results_target",
            "Conv6_FC3",
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--split",
            "0",
            "--transfer_learning_path",
            "results_source",
        ]
    elif request.param == "transfer_ae_cnn":
        source_task = [
            "train",
            "roi",
            "autoencoder",
            "data/dataset/random_example",
            "t1-linear",
            "data/labels_list",
            "results_source",
            "Conv4_FC3",
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--split",
            "0",
        ]
        target_task = [
            "train",
            "roi",
            "cnn",
            "data/dataset/random_example",
            "t1-linear",
            "data/labels_list",
            "results_target",
            "Conv4_FC3",
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--split",
            "0",
            "--transfer_learning_path",
            "results_source",
        ]
    elif request.param == "transfer_cnn_cnn":
        source_task = [
            "train",
            "roi",
            "cnn",
            "data/dataset/random_example",
            "t1-linear",
            "data/labels_list",
            "results_source",
            "Conv4_FC3",
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--split",
            "0",
        ]
        target_task = [
            "train",
            "roi",
            "cnn",
            "data/dataset/random_example",
            "t1-linear",
            "data/labels_list",
            "results_target",
            "Conv4_FC3",
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--split",
            "0",
            "--transfer_learning_path",
            "results_source",
        ]
    elif request.param == "transfer_cnn_multicnn":
        source_task = [
            "train",
            "patch",
            "cnn",
            "data/dataset/random_example",
            "t1-linear",
            "data/labels_list",
            "results_source",
            "Conv4_FC3",
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--split",
            "0",
        ]
        target_task = [
            "train",
            "patch",
            "multicnn",
            "data/dataset/random_example",
            "t1-linear",
            "data/labels_list",
            "results_target",
            "Conv4_FC3",
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--split",
            "0",
            "--transfer_learning_path",
            "results_source",
        ]
    else:
        raise NotImplementedError("Test %s is not implemented." % request.param)

    return source_task, target_task


def test_transfer(cli_commands):
    source_task, target_task = cli_commands
    flag_source = not os.system("clinicadl " + " ".join(source_task))
    flag_target = not os.system("clinicadl " + " ".join(target_task))
    assert flag_source
    assert flag_target
    shutil.rmtree("results_source")
    shutil.rmtree("results_target")
