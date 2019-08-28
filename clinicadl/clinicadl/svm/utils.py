from __future__ import print_function

diagnosis_code = {'CN': 0, 'AD': 1, 'sMCI': 0, 'pMCI': 1, 'MCI': 1, 'unlabeled': -1}


class SVMTester:
    def __init__(self, fold_dir):
        """
        :param fold_dir: Path to one fold of the trained SVM model.
        """
        import numpy as np
        from os import path
        from clinica.pipelines.machine_learning.voxel_based_io import revert_mask

        mask = np.loadtxt(path.join(fold_dir, 'data', 'train', 'mask.txt')).astype(bool)
        shape = np.loadtxt(path.join(fold_dir, 'data', 'train', 'shape.txt')).astype(int)

        weights = np.loadtxt(path.join(fold_dir, 'classifier', 'weights.txt'))
        self.weights = revert_mask(weights, mask, shape).flatten()
        self.intersect = np.loadtxt(path.join(fold_dir, 'classifier', 'intersect.txt'))

    def test(self, dataset):
        """
        :param dataset: (CAPSVoxelBasedInput) specific dataset of clinica initialized with test data.
        :return:
            (dict) metrics of evaluation
            (DataFrame) individual results of all sesssions
        """
        import numpy as np
        import pandas as pd

        images = dataset.get_x()
        labels = dataset.get_y()

        soft_prediction = np.dot(self.weights, images.transpose()) + self.intersect
        hard_prediction = (soft_prediction > 0).astype(int)
        subjects = dataset._subjects
        sessions = dataset._sessions
        data = np.array([subjects, sessions, labels, hard_prediction]).transpose()
        results_df = pd.DataFrame(data, columns=['participant_id', 'session_id', 'true_diagnosis', 'predicted_diagnosis'])

        return evaluate_prediction(labels, hard_prediction), results_df

    def test_and_save(self, dataset, evaluation_path):
        """
        :param dataset: (CAPSVoxelBasedInput) specific dataset of clinica initialized with test data.
        :param evaluation_path: (str) path to save the outputs
        :return: None
        """
        import pandas as pd
        import os

        metrics, results_df = self.test(dataset)
        metrics_df = pd.DataFrame(metrics, index=[0])

        if not os.path.exists(evaluation_path):
            os.makedirs(evaluation_path)
        metrics_df.to_csv(os.path.join(evaluation_path, 'metrics.tsv'), sep='\t', index=False)
        results_df.to_csv(os.path.join(evaluation_path, 'results.tsv'), sep='\t', index=False)


def evaluate_prediction(concat_true, concat_prediction, horizon=None):
    """
    This is a function to calculate the different metrics based on the list of true label and predicted label.

    :param concat_true: list of concatenated last labels
    :param concat_prediction: list of concatenated last prediction
    :param horizon: (int) number of batches to consider to evaluate performance
    :return: (dict) metrics
    """
    import numpy as np

    if horizon is not None:
        y = np.array(concat_true)[-horizon:]
        y_hat = np.array(concat_prediction)[-horizon:]
    else:
        y = np.array(concat_true)
        y_hat = np.array(concat_prediction)

    true_positive = np.sum((y_hat == 1) & (y == 1))
    true_negative = np.sum((y_hat == 0) & (y == 0))
    false_positive = np.sum((y_hat == 1) & (y == 0))
    false_negative = np.sum((y_hat == 0) & (y == 1))

    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

    if (true_positive + false_negative) != 0:
        sensitivity = true_positive / (true_positive + false_negative)
    else:
        sensitivity = 0.0

    if (false_positive + true_negative) != 0:
        specificity = true_negative / (false_positive + true_negative)
    else:
        specificity = 0.0

    if (true_positive + false_positive) != 0:
        ppv = true_positive / (true_positive + false_positive)
    else:
        ppv = 0.0

    if (true_negative + false_negative) != 0:
        npv = true_negative / (true_negative + false_negative)
    else:
        npv = 0.0

    balanced_accuracy = (sensitivity + specificity) / 2

    results = {'accuracy': accuracy,
               'balanced_accuracy': balanced_accuracy,
               'sensitivity': sensitivity,
               'specificity': specificity,
               'ppv': ppv,
               'npv': npv
               }

    return results


def save_data(df, output_dir, folder_name):
    """
    Save data so it can be used by the workflow

    :param df:
    :param output_dir:
    :param folder_name:
    :return: path to the tsv files
    """

    results_dir = path.join(output_dir, 'data', folder_name)
    if not path.exists(results_dir):
        os.makedirs(results_dir)

    df[['diagnosis']].to_csv(path.join(results_dir, 'diagnoses.tsv'), sep="\t", index=False)
    df[['participant_id', 'session_id']].to_csv(path.join(results_dir, 'sessions.tsv'), sep="\t", index=False)

    return results_dir


def save_additional_parameters(workflow, output_dir):
    """
    Saves additional parameters necessary for the testing phase (mask and original shape of the images).

    :param workflow: (MLWorkflow) workflow from which mask and original shape must be saved
    :return: None
    """
    import numpy as np

    mask = workflow._input._data_mask
    orig_shape = workflow._input._orig_shape
    np.savetxt(path.join(output_dir, 'mask.txt'), mask)
    np.savetxt(path.join(output_dir, 'shape.txt'), orig_shape)