# coding: utf8

import torch
import numpy as np
import os
import warnings
import pandas as pd
from time import time
import logging
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from sklearn.utils import column_or_1d
import scipy.sparse as sp

from clinicadl.tools.deep_learning.iotools import check_and_clean
from clinicadl.tools.deep_learning import EarlyStopping, save_checkpoint


#####################
# CNN train / test  #
#####################

def train(model, train_loader, valid_loader, criterion, optimizer, resume, log_dir, model_dir, options, logger=None):
    """
    Function used to train a CNN.
    The best model and checkpoint will be found in the 'best_model_dir' of options.output_dir.

    Args:
        model: (Module) CNN to be trained
        train_loader: (DataLoader) wrapper of the training dataset
        valid_loader: (DataLoader) wrapper of the validation dataset
        criterion: (loss) function to calculate the loss
        optimizer: (torch.optim) optimizer linked to model parameters
        resume: (bool) if True, a begun job is resumed
        log_dir: (str) path to the folder containing the logs
        model_dir: (str) path to the folder containing the models weights and biases
        options: (Namespace) ensemble of other options given to the main script.
        logger: (logging object) writer to stdout and stderr
    """
    from tensorboardX import SummaryWriter
    from time import time

    if logger is None:
        logger = logging

    columns = ['epoch', 'iteration', 'time',
               'balanced_accuracy_train', 'loss_train',
               'balanced_accuracy_valid', 'loss_valid']
    if hasattr(model, "variational") and model.variational:
        columns += ["kl_loss_train", "kl_loss_valid"]
    filename = os.path.join(os.path.dirname(log_dir), 'training.tsv')

    if not resume:
        check_and_clean(model_dir)
        check_and_clean(log_dir)

        results_df = pd.DataFrame(columns=columns)
        with open(filename, 'w') as f:
            results_df.to_csv(f, index=False, sep='\t')
        options.beginning_epoch = 0

    else:
        if not os.path.exists(filename):
            raise ValueError('The training.tsv file of the resumed experiment does not exist.')
        truncated_tsv = pd.read_csv(filename, sep='\t')
        truncated_tsv.set_index(['epoch', 'iteration'], inplace=True)
        truncated_tsv.drop(options.beginning_epoch, level=0, inplace=True)
        truncated_tsv.to_csv(filename, index=True, sep='\t')

    # Create writers
    writer_train = SummaryWriter(os.path.join(log_dir, 'train'))
    writer_valid = SummaryWriter(os.path.join(log_dir, 'validation'))

    # Initialize variables
    best_valid_accuracy = -1.0
    best_valid_loss = np.inf
    epoch = options.beginning_epoch

    model.train()  # set the model to training mode
    train_loader.dataset.train()

    early_stopping = EarlyStopping('min', min_delta=options.tolerance, patience=options.patience)
    mean_loss_valid = None
    t_beginning = time()

    while epoch < options.epochs and not early_stopping.step(mean_loss_valid):
        logger.info("Beginning epoch %i." % epoch)

        model.zero_grad()
        evaluation_flag = True
        step_flag = True
        tend = time()
        total_time = 0

        for i, data in enumerate(train_loader, 0):
            t0 = time()
            total_time = total_time + t0 - tend
            if options.gpu:
                imgs, labels = data['image'].cuda(), data['label'].cuda()
            else:
                imgs, labels = data['image'], data['label']

            if hasattr(model, "variational") and model.variational:
                z, mu, std, train_output = model(imgs)
                kl_loss = kl_divergence(z, mu, std)
                loss = criterion(train_output, labels) + kl_loss
            else:
                train_output = model(imgs)
                loss = criterion(train_output, labels)

            # Back propagation
            loss.backward()

            del imgs, labels

            if (i + 1) % options.accumulation_steps == 0:
                step_flag = False
                optimizer.step()
                optimizer.zero_grad()

                del loss

                # Evaluate the model only when no gradients are accumulated
                if options.evaluation_steps != 0 and (i + 1) % options.evaluation_steps == 0:
                    evaluation_flag = False

                    _, results_train = test(model, train_loader, options.gpu, criterion)
                    mean_loss_train = results_train["total_loss"] / (len(train_loader) * train_loader.batch_size)

                    _, results_valid = test(model, valid_loader, options.gpu, criterion)
                    mean_loss_valid = results_valid["total_loss"] / (len(valid_loader) * valid_loader.batch_size)
                    model.train()
                    train_loader.dataset.train()

                    global_step = i + epoch * len(train_loader)
                    writer_train.add_scalar('balanced_accuracy', results_train["balanced_accuracy"], global_step)
                    writer_train.add_scalar('loss', mean_loss_train, global_step)
                    writer_valid.add_scalar('balanced_accuracy', results_valid["balanced_accuracy"], global_step)
                    writer_valid.add_scalar('loss', mean_loss_valid, global_step)
                    logger.info("%s level training accuracy is %f at the end of iteration %d"
                                % (options.mode, results_train["balanced_accuracy"], i))
                    logger.info("%s level validation accuracy is %f at the end of iteration %d"
                                % (options.mode, results_valid["balanced_accuracy"], i))

                    t_current = time() - t_beginning
                    row = [epoch, i, t_current,
                           results_train["balanced_accuracy"], mean_loss_train,
                           results_valid["balanced_accuracy"], mean_loss_valid]
                    if hasattr(model, "variational") and model.variational:
                        row += [results_train["total_kl_loss"] / (len(train_loader) * train_loader.batch_size),
                                results_valid["total_kl_loss"] / (len(valid_loader) * valid_loader.batch_size)]
                    row_df = pd.DataFrame([row], columns=columns)
                    with open(filename, 'a') as f:
                        row_df.to_csv(f, header=False, index=False, sep='\t')

            tend = time()
        logger.debug('Mean time per batch loading: %.10f s'
                     % (total_time / len(train_loader) * train_loader.batch_size))

        # If no step has been performed, raise Exception
        if step_flag:
            raise Exception('The model has not been updated once in the epoch. The accumulation step may be too large.')

        # If no evaluation has been performed, warn the user
        elif evaluation_flag and options.evaluation_steps != 0:
            warnings.warn('Your evaluation steps are too big compared to the size of the dataset.'
                          'The model is evaluated only once at the end of the epoch')

        # Always test the results and save them once at the end of the epoch
        model.zero_grad()
        logger.debug('Last checkpoint at the end of the epoch %d' % epoch)

        _, results_train = test(model, train_loader, options.gpu, criterion)
        mean_loss_train = results_train["total_loss"] / (len(train_loader) * train_loader.batch_size)

        _, results_valid = test(model, valid_loader, options.gpu, criterion)
        mean_loss_valid = results_valid["total_loss"] / (len(valid_loader) * valid_loader.batch_size)
        model.train()
        train_loader.dataset.train()

        global_step = (epoch + 1) * len(train_loader)
        writer_train.add_scalar('balanced_accuracy', results_train["balanced_accuracy"], global_step)
        writer_train.add_scalar('loss', mean_loss_train, global_step)
        writer_valid.add_scalar('balanced_accuracy', results_valid["balanced_accuracy"], global_step)
        writer_valid.add_scalar('loss', mean_loss_valid, global_step)
        logger.info("%s level training accuracy is %f at the end of iteration %d"
                    % (options.mode, results_train["balanced_accuracy"], len(train_loader)))
        logger.info("%s level validation accuracy is %f at the end of iteration %d"
                    % (options.mode, results_valid["balanced_accuracy"], len(train_loader)))

        t_current = time() - t_beginning
        row = [epoch, i, t_current,
               results_train["balanced_accuracy"], mean_loss_train,
               results_valid["balanced_accuracy"], mean_loss_valid]
        if hasattr(model, "variational") and model.variational:
            row += [results_train["total_kl_loss"] / (len(train_loader) * train_loader.batch_size),
                    results_valid["total_kl_loss"] / (len(valid_loader) * valid_loader.batch_size)]
        row_df = pd.DataFrame([row], columns=columns)
        with open(filename, 'a') as f:
            row_df.to_csv(f, header=False, index=False, sep='\t')

        accuracy_is_best = results_valid["balanced_accuracy"] > best_valid_accuracy
        loss_is_best = mean_loss_valid < best_valid_loss
        best_valid_accuracy = max(results_valid["balanced_accuracy"], best_valid_accuracy)
        best_valid_loss = min(mean_loss_valid, best_valid_loss)

        save_checkpoint({'model': model.state_dict(),
                         'epoch': epoch,
                         'valid_loss': mean_loss_valid,
                         'valid_acc': results_valid["balanced_accuracy"]},
                        accuracy_is_best, loss_is_best,
                        model_dir)
        # Save optimizer state_dict to be able to reload
        save_checkpoint({'optimizer': optimizer.state_dict(),
                         'epoch': epoch,
                         'name': options.optimizer,
                         },
                        False, False,
                        model_dir,
                        filename='optimizer.pth.tar')

        epoch += 1

    os.remove(os.path.join(model_dir, "optimizer.pth.tar"))
    os.remove(os.path.join(model_dir, "checkpoint.pth.tar"))


def evaluate_prediction(y, y_pred):
    """
    Evaluates different metrics based on the list of true labels and predicted labels.

    Args:
        y: (list) true labels
        y_pred: (list) corresponding predictions

    Returns:
        (dict) ensemble of metrics
    """

    true_positive = np.sum((y_pred == 1) & (y == 1))
    true_negative = np.sum((y_pred == 0) & (y == 0))
    false_positive = np.sum((y_pred == 1) & (y == 0))
    false_negative = np.sum((y_pred == 0) & (y == 1))

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
               'npv': npv,
               }

    return results


def test(model, dataloader, use_cuda, criterion, mode="image", use_labels=True):
    """
    Computes the predictions and evaluation metrics.

    Args:
        model: (Module) CNN to be tested.
        dataloader: (DataLoader) wrapper of a dataset.
        use_cuda: (bool) if True a gpu is used.
        criterion: (loss) function to calculate the loss.
        mode: (str) input used by the network. Chosen from ['image', 'patch', 'roi', 'slice'].
        use_labels (bool): If True the true_label will be written in output DataFrame and metrics dict will be created.
    Returns
        (DataFrame) results of each input.
        (dict) ensemble of metrics + total loss on mode level.
    """
    model.eval()
    dataloader.dataset.eval()

    if mode == "image":
        columns = ["participant_id", "session_id", "true_label", "predicted_label"]
    elif mode in ["patch", "roi", "slice"]:
        columns = ['participant_id', 'session_id', '%s_id' % mode, 'true_label', 'predicted_label', 'proba0', 'proba1']
    else:
        raise ValueError("The mode %s is invalid." % mode)

    softmax = torch.nn.Softmax(dim=1)
    results_df = pd.DataFrame(columns=columns)
    total_loss = 0
    total_kl_loss = 0
    total_time = 0
    tend = time()
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            t0 = time()
            total_time = total_time + t0 - tend
            if use_cuda:
                inputs, labels = data['image'].cuda(), data['label'].cuda()
            else:
                inputs, labels = data['image'], data['label']

            if hasattr(model, "variational") and model.variational:
                z, mu, std, outputs = model(inputs)
                kl_loss = kl_divergence(z, mu, std)
                total_kl_loss += kl_loss.item()
            else:
                outputs = model(inputs)
            if use_labels:
                loss = criterion(outputs, labels)
                total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            # Generate detailed DataFrame
            for idx, sub in enumerate(data['participant_id']):
                if mode == "image":
                    row = [[sub, data['session_id'][idx], labels[idx].item(), predicted[idx].item()]]
                else:
                    normalized_output = softmax(outputs)
                    row = [[sub, data['session_id'][idx], data['%s_id' % mode][idx].item(),
                            labels[idx].item(), predicted[idx].item(),
                            normalized_output[idx, 0].item(), normalized_output[idx, 1].item()]]

                row_df = pd.DataFrame(row, columns=columns)
                results_df = pd.concat([results_df, row_df])

            del inputs, outputs, labels
            tend = time()
        results_df.reset_index(inplace=True, drop=True)

    if not use_labels:
        results_df = results_df.drop("true_label", axis=1)
        metrics_dict = None
    else:
        metrics_dict = evaluate_prediction(results_df.true_label.values.astype(int),
                                           results_df.predicted_label.values.astype(int))
        metrics_dict['total_loss'] = total_loss
        metrics_dict['total_kl_loss'] = total_kl_loss
    torch.cuda.empty_cache()

    return results_df, metrics_dict


def sort_predicted(model, data_df, input_dir, model_options, criterion, keep_true,
                   batch_size=1, num_workers=0, gpu=False):
    from .data import return_dataset, get_transforms
    from torch.utils.data import DataLoader
    from copy import copy

    if keep_true is None:
        return data_df

    _, all_transforms = get_transforms(model_options.mode, model_options.minmaxnormalization)
    dataset = return_dataset(mode=model_options.mode, input_dir=input_dir,
                             data_df=data_df, preprocessing=model_options.preprocessing,
                             train_transformations=None, all_transformations=all_transforms,
                             params=model_options)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True)

    test_options = copy(model_options)
    test_options.gpu = gpu

    results_df, _ = test(model, dataloader, gpu, criterion, model_options.mode, use_labels=True)

    sorted_df = data_df.sort_values(['participant_id', 'session_id']).reset_index(drop=True)
    results_df = results_df.sort_values(['participant_id', 'session_id']).reset_index(drop=True)

    if keep_true:
        return sorted_df[results_df.true_label == results_df.predicted_label].reset_index(drop=True)
    else:
        return sorted_df[results_df.true_label != results_df.predicted_label].reset_index(drop=True)


#################################
# Voting systems
#################################

def mode_level_to_tsvs(output_dir, results_df, metrics, fold, selection, mode, dataset='train', cnn_index=None):
    """
    Writes the outputs of the test function in tsv files.

    Args:
        output_dir: (str) path to the output directory.
        results_df: (DataFrame) the individual results per patch.
        metrics: (dict or DataFrame) the performances obtained on a series of metrics.
        fold: (int) the fold for which the performances were obtained.
        selection: (str) the metrics on which the model was selected (best_acc, best_loss)
        mode: (str) input used by the network. Chosen from ['image', 'patch', 'roi', 'slice'].
        dataset: (str) the dataset on which the evaluation was performed.
        cnn_index: (int) provide the cnn_index only for a multi-cnn framework.
    """
    if cnn_index is None:
        performance_dir = os.path.join(output_dir, 'fold-%i' % fold, 'cnn_classification', selection)
    else:
        performance_dir = os.path.join(output_dir, 'fold-%i' % fold, 'cnn_classification', 'cnn-%i' % cnn_index,
                                       selection)

    os.makedirs(performance_dir, exist_ok=True)

    results_df.to_csv(os.path.join(performance_dir, '%s_%s_level_prediction.tsv' % (dataset, mode)), index=False,
                      sep='\t')

    if metrics is not None:
        metrics["%s_id" % mode] = cnn_index
        if isinstance(metrics, dict):
            pd.DataFrame(metrics, index=[0]).to_csv(os.path.join(performance_dir, '%s_%s_level_metrics.tsv' % (dataset, mode)),
                                                    index=False, sep='\t')
        elif isinstance(metrics, pd.DataFrame):
            metrics.to_csv(os.path.join(performance_dir, '%s_%s_level_metrics.tsv' % (dataset, mode)),
                           index=False, sep='\t')
        else:
            raise ValueError("Bad type for metrics: %s. Must be dict or DataFrame." % type(metrics).__name__)


def concat_multi_cnn_results(output_dir, fold, selection, mode, dataset, num_cnn):
    """Concatenate the tsv files of a multi-CNN framework"""
    prediction_df = pd.DataFrame()
    metrics_df = pd.DataFrame()
    for cnn_index in range(num_cnn):
        cnn_dir = os.path.join(output_dir, 'fold-%i' % fold, 'cnn_classification', 'cnn-%i' % cnn_index)
        performance_dir = os.path.join(cnn_dir, selection)
        cnn_pred_path = os.path.join(performance_dir, '%s_%s_level_prediction.tsv' % (dataset, mode))
        cnn_metrics_path = os.path.join(performance_dir, '%s_%s_level_metrics.tsv' % (dataset, mode))

        cnn_pred_df = pd.read_csv(cnn_pred_path, sep='\t')
        prediction_df = pd.concat([prediction_df, cnn_pred_df])
        os.remove(cnn_pred_path)

        if os.path.exists(cnn_metrics_path):
            cnn_metrics_df = pd.read_csv(cnn_metrics_path, sep='\t')
            metrics_df = pd.concat([metrics_df, cnn_metrics_df])
            os.remove(cnn_metrics_path)

        # Clean unused files
        if len(os.listdir(performance_dir)) == 0:
            os.rmdir(performance_dir)
        if len(os.listdir(cnn_dir)) == 0:
            os.rmdir(cnn_dir)

    prediction_df.reset_index(drop=True, inplace=True)
    if len(metrics_df) == 0:
        metrics_df = None
    else:
        metrics_df.reset_index(drop=True, inplace=True)
    mode_level_to_tsvs(output_dir, prediction_df, metrics_df, fold, selection, mode, dataset)


def retrieve_sub_level_results(output_dir, fold, selection, mode, dataset, num_cnn):
    """Retrieve performance_df for single or multi-CNN framework.
    If the results of the multi-CNN were not concatenated it will be done here."""
    result_tsv = os.path.join(output_dir, 'fold-%i' % fold, 'cnn_classification', selection,
                              '%s_%s_level_prediction.tsv' % (dataset, mode))
    if os.path.exists(result_tsv):
        performance_df = pd.read_csv(result_tsv, sep='\t')

    else:
        concat_multi_cnn_results(output_dir, fold, selection, mode, dataset, num_cnn)
        performance_df = pd.read_csv(result_tsv, sep='\t')

    return performance_df


def soft_voting_to_tsvs(output_dir, fold, selection, mode, dataset='test', num_cnn=None,
                        selection_threshold=None, logger=None, use_labels=True):
    """
    Writes soft voting results in tsv files.

    Args:
        output_dir: (str) path to the output directory.
        fold: (int) Fold number of the cross-validation.
        selection: (str) criterion on which the model is selected (either best_loss or best_acc)
        mode: (str) input used by the network. Chosen from ['patch', 'roi', 'slice'].
        dataset: (str) name of the dataset for which the soft-voting is performed. If different from training or
            validation, the weights of soft voting will be computed on validation accuracies.
        num_cnn: (int) if given load the patch level results of a multi-CNN framework.
        selection_threshold: (float) all patches for which the classification accuracy is below the
            threshold is removed.
        logger: (logging object) writer to stdout and stderr
        use_labels: (bool) If True the labels are added to the final tsv
    """
    if logger is None:
        logger = logging

    # Choose which dataset is used to compute the weights of soft voting.
    if dataset in ['train', 'validation']:
        validation_dataset = dataset
    else:
        validation_dataset = 'validation'
    test_df = retrieve_sub_level_results(output_dir, fold, selection, mode, dataset, num_cnn)
    validation_df = retrieve_sub_level_results(output_dir, fold, selection, mode, validation_dataset, num_cnn)

    performance_path = os.path.join(output_dir, 'fold-%i' % fold, 'cnn_classification', selection)
    os.makedirs(performance_path, exist_ok=True)

    df_final, metrics = soft_voting(test_df, validation_df, mode, selection_threshold=selection_threshold,
                                    use_labels=use_labels)

    df_final.to_csv(os.path.join(os.path.join(performance_path, '%s_image_level_prediction.tsv' % dataset)),
                    index=False, sep='\t')
    if use_labels:
        pd.DataFrame(metrics, index=[0]).to_csv(os.path.join(performance_path, '%s_image_level_metrics.tsv' % dataset),
                                                index=False, sep='\t')
        logger.info("image level %s balanced accuracy is %f for model selected on %s"
                    % (dataset, metrics["balanced_accuracy"], selection))


def soft_voting(performance_df, validation_df, mode, selection_threshold=None, use_labels=True):
    """
    Computes soft voting based on the probabilities in performance_df. Weights are computed based on the accuracies
    of validation_df.

    ref: S. Raschka. Python Machine Learning., 2015

    Args:
        performance_df: (DataFrame) results on patch level of the set on which the combination is made.
        validation_df: (DataFrame) results on patch level of the set used to compute the weights.
        mode: (str) input used by the network. Chosen from ['patch', 'roi', 'slice'].
        selection_threshold: (float) if given, all patches for which the classification accuracy is below the
            threshold is removed.

    Returns:
        df_final (DataFrame) the results on the image level
        results (dict) the metrics on the image level
    """

    # Compute the sub-level accuracies on the validation set:
    validation_df["accurate_prediction"] = validation_df.apply(lambda x: check_prediction(x), axis=1)
    sub_level_accuracies = validation_df.groupby("%s_id" % mode)["accurate_prediction"].sum()
    if selection_threshold is not None:
        sub_level_accuracies[sub_level_accuracies < selection_threshold] = 0
    weight_series = sub_level_accuracies / sub_level_accuracies.sum()

    # Sort to allow weighted average computation
    performance_df.sort_values(['participant_id', 'session_id', '%s_id' % mode], inplace=True)
    weight_series.sort_index(inplace=True)

    # Soft majority vote
    if use_labels:
        columns = ['participant_id', 'session_id', 'true_label', 'predicted_label']
    else:
        columns = ['participant_id', 'session_id', 'predicted_label']
    df_final = pd.DataFrame(columns=columns)
    for (subject, session), subject_df in performance_df.groupby(['participant_id', 'session_id']):
        proba0 = np.average(subject_df["proba0"], weights=weight_series)
        proba1 = np.average(subject_df["proba1"], weights=weight_series)
        proba_list = [proba0, proba1]
        y_hat = proba_list.index(max(proba_list))

        if use_labels:
            y = subject_df["true_label"].unique().item()
            row = [[subject, session, y, y_hat]]
        else:
            row = [[subject, session, y_hat]]
        row_df = pd.DataFrame(row, columns=columns)
        df_final = df_final.append(row_df)

    if use_labels:
        results = evaluate_prediction(df_final.true_label.values.astype(int),
                                      df_final.predicted_label.values.astype(int))
    else:
        results = None

    return df_final, results


def mode_to_image_tsvs(output_dir, fold, selection, mode, dataset="test"):
    """
    Copy mode-level tsvs to name them as image-level tsvs
    Args:
        output_dir: (str) path to the output directory.
        fold: (int) Fold number of the cross-validation.
        selection: (str) criterion on which the model is selected (either best_loss or best_acc)
        mode: (str) input used by the network. Chosen from ['patch', 'roi', 'slice'].
        dataset: (str) name of the dataset for which the soft-voting is performed. If different from training or
            validation, the weights of soft voting will be computed on validation accuracies.
    """
    sub_df = retrieve_sub_level_results(output_dir, fold, selection, mode, dataset, 1)
    sub_df.drop([f'{mode}_id', 'proba0', 'proba1'], axis=1, inplace=True)

    performance_path = os.path.join(output_dir, f'fold-{fold}', 'cnn_classification', selection)
    sub_df.to_csv(os.path.join(performance_path, f'{dataset}_image_level_prediction.tsv'),
                  index=False, sep='\t')
    metrics_df = pd.read_csv(os.path.join(performance_path, f'{dataset}_{mode}_level_metrics.tsv'), sep="\t")
    metrics_df.drop([f'{mode}_id'], axis=1, inplace=True)
    metrics_df.to_csv(os.path.join(performance_path, f'{dataset}_image_level_metrics.tsv'),
                      index=False, sep='\t')


def check_prediction(row):
    if row["true_label"] == row["predicted_label"]:
        return 1
    else:
        return 0


###########
# Loss
###########
class L1ClassificationLoss(_Loss):
    def __init__(self, reduction="mean", normalization=True, n_classes=2):
        super(L1ClassificationLoss, self).__init__(reduction=reduction)
        self.softmax = torch.nn.Softmax(dim=1)
        self.normalization = normalization
        self.classes = np.arange(n_classes)

    def forward(self, input, target):
        if self.normalization:
            input = self.softmax(input)
        binarize_target = binarize_label(target, self.classes)
        return F.l1_loss(input, binarize_target)


class SmoothL1ClassificationLoss(_Loss):
    def __init__(self, reduction="mean", normalization=True, n_classes=2):
        super(SmoothL1ClassificationLoss, self).__init__(reduction=reduction)
        self.softmax = torch.nn.Softmax(dim=1)
        self.normalization = normalization
        self.classes = np.arange(n_classes)

    def forward(self, input, target):
        if self.normalization:
            input = self.softmax(input)
        binarize_target = binarize_label(target, self.classes)
        return F.smooth_l1_loss(input, binarize_target)


def get_criterion(option):
    """Returns the appropriate loss depending on the option"""
    if option == "default":
        return torch.nn.CrossEntropyLoss(reduction="sum")
    elif option == "L1Norm" or option == "L1":
        return L1ClassificationLoss(reduction="sum", normalization=(option == "L1Norm"))
    elif option == "SmoothL1Norm" or option == "SmoothL1":
        return SmoothL1ClassificationLoss(reduction="sum", normalization=(option == "SmoothL1Norm"))
    else:
        raise ValueError("The option %s is unknown for criterion selection" % option)


def binarize_label(y, classes, pos_label=1, neg_label=0):
    """Greatly inspired from scikit-learn"""
    sorted_class = np.sort(classes)
    device = y.device
    y = y.cpu().numpy()
    n_samples = len(y)
    n_classes = len(classes)
    y = column_or_1d(y)

    # pick out the known labels from y
    y_in_classes = np.in1d(y, classes)
    y_seen = y[y_in_classes]
    indices = np.searchsorted(sorted_class, y_seen)
    indptr = np.hstack((0, np.cumsum(y_in_classes)))

    data = np.empty_like(indices)
    data.fill(pos_label)
    Y = sp.csr_matrix((data, indices, indptr),
                      shape=(n_samples, n_classes))
    Y = Y.toarray()
    Y = Y.astype(int, copy=False)
    if neg_label != 0:
        Y[Y == 0] = neg_label
    Y = torch.from_numpy(Y).float().to(device)
    return Y


def kl_divergence(z, mu, std):
    """
    Monte-Carlo KL divergence from
    https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed

    Args:
        z: The vector sampled from Normal(mu, std) in the variational network
        mu: The mean value of the Normal distribution of the variational network
        std: The standard deviation of the Normal distribution of the variational network
    Returns:
        The value of the KL divergence between the Normal distributions of mean 0 and std 1 and of mean mu and std std.
    """
    # 1. define the first two probabilities (in this case Normal for both)
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)

    # 2. get the probabilities from the equation
    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)

    # kl
    kl = (log_qzx - log_pz)

    # go from single dim distribution to multi-dim
    kl = kl.mean(-1).sum()
    return kl
