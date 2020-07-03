# coding: utf8

import torch
import numpy as np
import os
import warnings
import pandas as pd
from time import time

from clinicadl.tools.deep_learning.iotools import check_and_clean
from clinicadl.tools.deep_learning import EarlyStopping, save_checkpoint


#####################
# CNN train / test  #
#####################

def train(model, train_loader, valid_loader, criterion, optimizer, resume, log_dir, model_dir, options):
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
    """
    from tensorboardX import SummaryWriter
    from time import time

    if not resume:
        check_and_clean(model_dir)
        check_and_clean(log_dir)

    # Create writers
    writer_train = SummaryWriter(os.path.join(log_dir, 'train'))
    writer_valid = SummaryWriter(os.path.join(log_dir, 'validation'))

    # Initialize variables
    best_valid_accuracy = 0.0
    best_valid_loss = np.inf
    epoch = options.beginning_epoch

    model.train()  # set the module to training mode

    early_stopping = EarlyStopping('min', min_delta=options.tolerance, patience=options.patience)
    mean_loss_valid = None

    while epoch < options.epochs and not early_stopping.step(mean_loss_valid):
        print("At %d-th epoch." % epoch)

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
            train_output = model(imgs)
            _, predict_batch = train_output.topk(1)
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
                    print('Iteration %d' % i)

                    _, results_train = test(model, train_loader, options.gpu, criterion)
                    mean_loss_train = results_train["total_loss"] / (len(train_loader) * train_loader.batch_size)

                    _, results_valid = test(model, valid_loader, options.gpu, criterion)
                    mean_loss_valid = results_valid["total_loss"] / (len(valid_loader) * valid_loader.batch_size)
                    model.train()

                    global_step = i + epoch * len(train_loader)
                    writer_train.add_scalar('balanced_accuracy', results_train["balanced_accuracy"], global_step)
                    writer_train.add_scalar('loss', mean_loss_train, global_step)
                    writer_valid.add_scalar('balanced_accuracy', results_valid["balanced_accuracy"], global_step)
                    writer_valid.add_scalar('loss', mean_loss_valid, global_step)
                    print("%s level training accuracy is %f at the end of iteration %d"
                          % (options.mode, results_train["balanced_accuracy"], i))
                    print("%s level validation accuracy is %f at the end of iteration %d"
                          % (options.mode, results_valid["balanced_accuracy"], i))

            tend = time()
        print('Mean time per batch loading (train):', total_time / len(train_loader) * train_loader.batch_size)

        # If no step has been performed, raise Exception
        if step_flag:
            raise Exception('The model has not been updated once in the epoch. The accumulation step may be too large.')

        # If no evaluation has been performed, warn the user
        elif evaluation_flag and options.evaluation_steps != 0:
            warnings.warn('Your evaluation steps are too big compared to the size of the dataset.'
                          'The model is evaluated only once at the end of the epoch')

        # Always test the results and save them once at the end of the epoch
        model.zero_grad()
        print('Last checkpoint at the end of the epoch %d' % epoch)

        _, results_train = test(model, train_loader, options.gpu, criterion)
        mean_loss_train = results_train["total_loss"] / (len(train_loader) * train_loader.batch_size)

        _, results_valid = test(model, valid_loader, options.gpu, criterion)
        mean_loss_valid = results_valid["total_loss"] / (len(valid_loader) * valid_loader.batch_size)
        model.train()

        global_step = (epoch + 1) * len(train_loader)
        writer_train.add_scalar('balanced_accuracy', results_train["balanced_accuracy"], global_step)
        writer_train.add_scalar('loss', mean_loss_train, global_step)
        writer_valid.add_scalar('balanced_accuracy', results_valid["balanced_accuracy"], global_step)
        writer_valid.add_scalar('loss', mean_loss_valid, global_step)
        print("%s level training accuracy is %f at the end of iteration %d"
              % (options.mode, results_train["balanced_accuracy"], len(train_loader)))
        print("%s level validation accuracy is %f at the end of iteration %d"
              % (options.mode, results_valid["balanced_accuracy"], len(train_loader)))

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


def test(model, dataloader, use_cuda, criterion, mode="image"):
    """
    Computes the predictions and evaluation metrics.

    Args:
        model: (Module) CNN to be tested.
        dataloader: (DataLoader) wrapper of a dataset.
        use_cuda: (bool) if True a gpu is used.
        criterion: (loss) function to calculate the loss.
        mode: (str) input used by the network. Chosen from ['image', 'patch', 'roi', 'slice'].
    Returns
        (DataFrame) results of each input.
        (dict) ensemble of metrics + total loss on mode level.
    """
    model.eval()

    if mode == "image":
        columns = ["participant_id", "session_id", "true_label", "predicted_label"]
    elif mode in ["patch", "roi", "slice"]:
        columns = ['participant_id', 'session_id', '%s_id' % mode, 'true_label', 'predicted_label', 'proba0', 'proba1']
    else:
        raise ValueError("The mode %s is invalid." % mode)

    softmax = torch.nn.Softmax(dim=1)
    results_df = pd.DataFrame(columns=columns)
    total_loss = 0
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
            outputs = model(inputs)
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

            del inputs, outputs, labels, loss
            tend = time()
        print('Mean time per batch loading (test):', total_time / len(dataloader) * dataloader.batch_size)
        results_df.reset_index(inplace=True, drop=True)

        # calculate the balanced accuracy
        results = evaluate_prediction(results_df.true_label.values.astype(int),
                                      results_df.predicted_label.values.astype(int))
        results_df.reset_index(inplace=True, drop=True)
        results['total_loss'] = total_loss
        torch.cuda.empty_cache()

    return results_df, results


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
        metrics["%s_id" % mode] = cnn_index

    if not os.path.exists(performance_dir):
        os.makedirs(performance_dir)

    results_df.to_csv(os.path.join(performance_dir, '%s_%s_level_prediction.tsv' % (dataset, mode)), index=False,
                      sep='\t')

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
        cnn_metrics_df = pd.read_csv(cnn_metrics_path, sep='\t')
        prediction_df = pd.concat([prediction_df, cnn_pred_df])
        metrics_df = pd.concat([metrics_df, cnn_metrics_df])

        # Clean unused files
        os.remove(cnn_pred_path)
        os.remove(cnn_metrics_path)
        if len(os.listdir(performance_dir)) == 0:
            os.rmdir(performance_dir)
        if len(os.listdir(cnn_dir)) == 0:
            os.rmdir(cnn_dir)

    prediction_df.reset_index(drop=True, inplace=True)
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


def soft_voting_to_tsvs(output_dir, fold, selection, mode, dataset='test', num_cnn=None, selection_threshold=None):
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
    """

    # Choose which dataset is used to compute the weights of soft voting.
    if dataset in ['train', 'validation']:
        validation_dataset = dataset
    else:
        validation_dataset = 'validation'
    test_df = retrieve_sub_level_results(output_dir, fold, selection, mode, dataset, num_cnn)
    validation_df = retrieve_sub_level_results(output_dir, fold, selection, mode, validation_dataset, num_cnn)

    performance_path = os.path.join(output_dir, 'fold-%i' % fold, 'cnn_classification', selection)
    if not os.path.exists(performance_path):
        os.makedirs(performance_path)

    df_final, metrics = soft_voting(test_df, validation_df, mode, selection_threshold=selection_threshold)

    df_final.to_csv(os.path.join(os.path.join(performance_path, '%s_image_level_prediction.tsv' % dataset)),
                    index=False, sep='\t')

    pd.DataFrame(metrics, index=[0]).to_csv(os.path.join(performance_path, '%s_image_level_metrics.tsv' % dataset),
                                            index=False, sep='\t')


def soft_voting(performance_df, validation_df, mode, selection_threshold=None):
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
    columns = ['participant_id', 'session_id', 'true_label', 'predicted_label']
    df_final = pd.DataFrame(columns=columns)
    for (subject, session), subject_df in performance_df.groupby(['participant_id', 'session_id']):
        y = subject_df["true_label"].unique().item()
        proba0 = np.average(subject_df["proba0"], weights=weight_series)
        proba1 = np.average(subject_df["proba1"], weights=weight_series)
        proba_list = [proba0, proba1]
        y_hat = proba_list.index(max(proba_list))

        row = [[subject, session, y, y_hat]]
        row_df = pd.DataFrame(row, columns=columns)
        df_final = df_final.append(row_df)

    results = evaluate_prediction(df_final.true_label.values.astype(int),
                                  df_final.predicted_label.values.astype(int))

    return df_final, results


def check_prediction(row):
    if row["true_label"] == row["predicted_label"]:
        return 1
    else:
        return 0
