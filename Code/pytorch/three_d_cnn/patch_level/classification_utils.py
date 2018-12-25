import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import shutil
import warnings
import pandas as pd
from time import time
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import os, shutil
from skimage.transform import resize
from os import path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

def train(model, data_loader, use_cuda, loss_func, optimizer, writer, epoch_i, model_mode="train", global_steps=0):
    """
    This is the function to train, validate or test the model, depending on the model_mode parameter.
    :param model:
    :param data_loader:
    :param use_cuda:
    :param loss_func:
    :param optimizer:
    :param writer:
    :param epoch_i:
    :return:
    """
    # main training loop
    correct_cnt = 0.0
    acc = 0.0
    subjects = []
    y_ground = []
    y_hat = []

    if model_mode == "train":
        model.train() ## set the model to training mode
    else:
        model.eval() ## set the model to evaluation mode
    print('The number of batches in this sampler based on the batch size: %s' % str(len(data_loader)))
    for i, subject_data in enumerate(data_loader):
        # for each iteration, the train data contains batch_size * n_patchs_in_each_subject images
        loss_batch = 0.0
        acc_batch = 0.0
        num_patch = len(subject_data)

        print('The number of patchs in one subject is: %s' % str(num_patch))

        for j in range(num_patch):
            data_dic = subject_data[j]
            if use_cuda:
                imgs, labels = data_dic['image'].cuda(), data_dic['label'].cuda()
            else:
                imgs, labels = data_dic['image'], data_dic['label']

            ## add the participant_id + session_id
            image_ids = data_dic['image_id']
            subjects.extend(image_ids)

            # TO track of indices, int64 is a better choice for large models.
            integer_encoded = labels.data.cpu().numpy()
            gound_truth_list = integer_encoded.tolist()
            y_ground.extend(gound_truth_list)
            ground_truth = Variable(torch.from_numpy(integer_encoded)).long()

            print('The group true label is %s' % (str(labels)))
            if use_cuda:
                ground_truth = ground_truth.cuda()
            output = model(imgs)
            _, predict = output.topk(1)
            predict_list = predict.data.cpu().numpy().tolist()
            y_hat.extend([item for sublist in predict_list for item in sublist])
            if model_mode == "train" or model_mode == 'valid':
                print("output.device: " + str(output.device))
                print("ground_truth.device: " + str(ground_truth.device))
                print("The predicted label is: " + str(output))
                loss = loss_func(output, ground_truth)
                loss_batch += loss.item()
            correct_this_batch = (predict.squeeze(1) == ground_truth).sum().float()
            correct_cnt += correct_this_batch
            # To monitor the training process using tensorboard, we only display the training loss and accuracy, the other performance metrics, such
            # as balanced accuracy, will be saved in the tsv file.
            accuracy = float(correct_this_batch) / len(ground_truth)
            acc_batch += accuracy
            if model_mode == "train":
                print("For batch %d patch %d training loss is : %f" % (i, j, loss.item()))
                print("For batch %d patch %d training accuracy is : %f" % (i, j, accuracy))
            elif model_mode == "valid":
                print("For batch %d patch %d validation accuracy is : %f" % (i, j, accuracy))
                print("For batch %d patch %d validation loss is : %f" % (i, j, loss.item()))
            elif model_mode == "test":
                print("For batch %d patch %d validate accuracy is : %f" % (i, j, accuracy))

            # Unlike tensorflow, in Pytorch, we need to manully zero the graident before each backpropagation step, becase Pytorch accumulates the gradients
            # on subsequent backward passes. The initial designing for this is convenient for training RNNs.
            if model_mode == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # delete the temporal varibles taking the GPU memory
            if i == 0 and j == 0:
                example_imgs = imgs[:, :, 1, :, :]
            del imgs, labels, output, ground_truth, loss, predict

        if model_mode == "train":
            writer.add_scalar('patch-level accuracy', acc_batch / num_patch, i + epoch_i * len(data_loader.dataset))
            writer.add_scalar('loss', loss_batch / num_patch, i + epoch_i * len(data_loader.dataset))
            ## just for debug
            writer.add_image('example_image', example_imgs)
        elif model_mode == "test":
            writer.add_scalar('patch-level accuracy', acc_batch / num_patch, i)

        ## add all accuracy for each iteration
        acc += acc_batch / num_patch

    acc_mean = acc / len(data_loader)
    if model_mode == "valid":
        writer.add_scalar('patch-level accuracy', acc_mean, global_steps)
        writer.add_scalar('loss', loss_batch / num_patch / i, global_steps)

    if model_mode == "train":
        global_steps = i + epoch_i * len(data_loader.dataset)
    else:
        global_steps = 0

    return example_imgs, subjects, y_ground, y_hat, acc_mean, global_steps


def train_ae(autoencoder, data_loader, use_cuda, loss_func, optimizer, writer, epoch_i, options, global_steps=0):
    """
    This trains the autoencoder with all data
    :param autoencoder:
    :param data_loader:
    :param use_cuda:
    :param loss_func:
    :param optimizer:
    :param writer:
    :param epoch_i:
    :param global_steps:
    :return:
    """
    epoch_loss = 0
    sparsity = 0.05
    beta = 0.5
    print('The number of batches in this sampler based on the batch size: %s' % str(len(data_loader)))
    for i, subject_data in enumerate(data_loader):
        # for each iteration, the train data contains batch_size * n_patchs_in_each_subject images
        loss_batch = 0.0
        num_patch = len(subject_data)

        print('The number of patchs in one subject is: %s' % str(num_patch))

        for j in range(num_patch):
            data_dic = subject_data[j]
            if use_cuda:
                imgs = data_dic['image'].cuda()
            else:
                imgs = data_dic['image']

            output, hidden = autoencoder(imgs)
            loss1 = loss_func(output, imgs)
            sparsity_part = Variable(torch.ones(hidden.shape) * sparsity).cuda()
            loss2 = (sparsity_part * torch.log(sparsity_part / (hidden + 1e-8)) + (1 - sparsity_part) * torch.log(
                (1 - sparsity_part) / ((1 - hidden + 1e-8)))).sum() / options.batch_size
            # kl_div_loss(mean_activitaion, sparsity)
            loss = loss1 + beta * loss2
            loss_batch += loss
            print("For batch %d patch %d training loss is : %f" % (i, j, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i == 0 and j == 0:
                example_imgs = imgs[:, :, 1, :, :]
            ## save memory
            del imgs, output, loss

        ## save loss into tensorboardX
        writer.add_scalar('loss', loss_batch / num_patch, i + epoch_i * len(data_loader.dataset))
        epoch_loss += loss_batch

    return example_imgs, epoch_loss



def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2) # put ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(checkpoint_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(checkpoint_dir, filename),  os.path.join(checkpoint_dir, 'model_best.pth.tar'))


def load_model(model, checkpoint_dir, filename='model_best.pth.tar'):
    from copy import deepcopy

    best_model = deepcopy(model)
    param_dict = torch.load(os.path.join(checkpoint_dir, filename))
    best_model.load_state_dict(param_dict['model'])
    return best_model, param_dict['epoch']


def check_and_clean(d):

    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)


def ae_finetuning(decoder, train_loader, valid_loader, loss_func, gpu, results_path, options):
    from os import path

    if not path.exists(results_path):
        os.makedirs(results_path)
    filename = os.path.join(results_path, 'training.tsv')

    columns = ['epoch', 'iteration', 'loss_train', 'mean_loass_train', 'loss_valid', 'mean_loss_valid']
    results_df = pd.DataFrame(columns=columns)
    with open(filename, 'w') as f:
        results_df.to_csv(f, index=False, sep='\t')

    decoder.train()
    optimizer = eval("torch.optim." + options.optimizer)(filter(lambda x: x.requires_grad, decoder.parameters()),
                                                         options.transfer_learning_rate)
    print(decoder)

    if gpu:
        decoder.cuda()

    # Initialize variables
    best_loss_valid = np.inf
    print("Beginning training")
    for epoch in range(options.transfer_learning_epochs):
        print("At %d-th epoch." % epoch)

        decoder.zero_grad()
        evaluation_flag = True
        step_flag = True
        last_check_point_i = 0
        for i, data in enumerate(train_loader):
            if gpu:
                imgs = data['image'].cuda()
            else:
                imgs = data['image']

            train_output = decoder(imgs)
            loss = loss_func(train_output, imgs)
            loss.backward()

            # writer_train.add_scalar('training_loss', loss.item() / len(data), i + epoch * len(train_loader.dataset))

            if (i+1) % options.accumulation_steps == 0:
                step_flag = False
                optimizer.step()
                optimizer.zero_grad()

                # Evaluate the decoder only when no gradients are accumulated
                if (i+1) % options.evaluation_steps == 0:
                    evaluation_flag = False
                    print('Iteration %d' % i)
                    loss_train = test_ae(decoder, train_loader, gpu, loss_func)
                    mean_loss_train = loss_train / (len(train_loader) * train_loader.dataset.size)
                    loss_valid = test_ae(decoder, valid_loader, gpu, loss_func)
                    mean_loss_valid = loss_valid / (len(valid_loader) * valid_loader.dataset.size)
                    decoder.train()
                    print("Scan level validation loss is %f at the end of iteration %d" % (loss_valid, i))
                    row = np.array([epoch, i, loss_train, mean_loss_train, loss_valid, mean_loss_valid]).reshape(1, -1)
                    row_df = pd.DataFrame(row, columns=columns)
                    with open(filename, 'a') as f:
                        row_df.to_csv(f, header=False, index=False, sep='\t')

            del imgs, train_output

        # If no step has been performed, raise Exception
        if step_flag:
            raise Exception('The model has not been updated once in the epoch. The accumulation step may be too large.')

        # If no evaluation has been performed, warn the user
        if evaluation_flag:
            warnings.warn('Your evaluation steps are too big compared to the size of the dataset.'
                          'The model is evaluated only once at the end of the epoch')

        # Always test the results and save them once at the end of the epoch
        if last_check_point_i != i:
            print('Last checkpoint at the end of the epoch %d' % epoch)
            loss_train = test_ae(decoder, train_loader, gpu, loss_func)
            mean_loss_train = loss_train / (len(train_loader) * train_loader.dataset.size)
            loss_valid = test_ae(decoder, valid_loader, gpu, loss_func)
            mean_loss_valid = loss_valid / (len(valid_loader) * valid_loader.dataset.size)
            decoder.train()
            print("Scan level validation loss is %f at the end of iteration %d" % (loss_valid, i))

            row = np.array([epoch, i, loss_train, mean_loss_train, loss_valid, mean_loss_valid]).reshape(1, -1)
            row_df = pd.DataFrame(row, columns=columns)
            with open(filename, 'a') as f:
                row_df.to_csv(f, header=False, index=False, sep='\t')

            is_best = loss_valid < best_loss_valid
            # Save only if is best to avoid performance deterioration
            if is_best:
                best_loss_valid = loss_valid
                save_checkpoint({'model': decoder.state_dict(),
                                 'iteration': i,
                                 'epoch': epoch,
                                 'loss_valid': loss_valid},
                                is_best,
                                results_path)

    # print('End of training', torch.cuda.memory_allocated())


def test_ae(model, dataloader, use_cuda, loss_func, first_layers=None):
    """
    Computes the loss of the model

    :param model: the network (subclass of nn.Module)
    :param dataloader: a DataLoader wrapping a dataset
    :param use_cuda: if True a gpu is used
    :return: loss of the model (float)
    """
    model.eval()

    total_loss = 0
    for i, data in enumerate(dataloader, 0):
        if use_cuda:
            inputs = data['image'].cuda()
        else:
            inputs = data['image']

        if first_layers is not None:
            hidden = first_layers(inputs)
        else:
            hidden = inputs
        outputs = model(hidden)
        loss = loss_func(outputs, hidden)
        total_loss += loss.item()

        del inputs, outputs, loss

    return total_loss


def greedy_learning(model, train_loader, valid_loader, loss_func, gpu, results_path, options):
    from os import path
    from model import Decoder
    from copy import deepcopy

    decoder = Decoder(model)

    level = 0
    first_layers = extract_first_layers(decoder, level)
    auto_encoder = extract_ae(decoder, level)

    while len(auto_encoder) > 0:
        print('Cell learning level %i' % level)
        level_path = path.join(results_path, 'level-' + str(level))
        # Create the method to train with first layers
        ae_training(auto_encoder, first_layers, train_loader, valid_loader, loss_func, gpu, level_path, options)
        best_ae, _ = load_model(auto_encoder, level_path)

        # Copy the weights of best_ae in decoder encoder and decoder layers
        set_weights(decoder, best_ae, level)

        # Prepare next iteration
        level += 1
        first_layers = extract_first_layers(decoder, level)
        auto_encoder = extract_ae(decoder, level)

    ae_finetuning(decoder, train_loader, valid_loader, loss_func, gpu, results_path, options)

    # Updating and setting weights of the convolutional layers
    best_decoder, best_epoch = load_model(decoder, results_path)
    model.features = deepcopy(best_decoder.encoder)
    save_checkpoint({'model': model.state_dict(),
                     'epoch': best_epoch},
                    False,
                    os.path.join(results_path),
                    'model_pretrained.pth.tar')

    if options.visualization:
        visualize_ae(best_decoder, train_loader, os.path.join(results_path, "train"), gpu)
        visualize_ae(best_decoder, valid_loader, os.path.join(results_path, "valid"), gpu)

    return model


def set_weights(decoder, auto_encoder, level):
    import torch.nn as nn

    n_conv = 0
    i_ae = 0

    for i, layer in enumerate(decoder.encoder):
        if isinstance(layer, nn.Conv3d):
            n_conv += 1

        if n_conv == level + 1:
            decoder.encoder[i] = auto_encoder.encoder[i_ae]
            # Do BatchNorm layers are not used in decoder
            if not isinstance(layer, nn.BatchNorm3d):
                decoder.decoder[len(decoder) - (i+1)] = auto_encoder.decoder[len(auto_encoder) - (i_ae+1)]
            i_ae += 1

    return decoder


def ae_training(auto_encoder, first_layers, train_loader, valid_loader, loss_func, gpu, results_path, options):
    from os import path

    if not path.exists(results_path):
        os.makedirs(results_path)

    filename = os.path.join(results_path, 'training.tsv')
    columns = ['epoch', 'iteration', 'loss_train', 'mean_loass_train', 'loss_valid', 'mean_loss_valid']
    results_df = pd.DataFrame(columns=columns)
    with open(filename, 'w') as f:
        results_df.to_csv(f, index=False, sep='\t')

    auto_encoder.train()
    first_layers.eval()
    print(first_layers)
    print(auto_encoder)
    optimizer = eval("torch.optim." + options.optimizer)(filter(lambda x: x.requires_grad, auto_encoder.parameters()),
                                                         options.transfer_learning_rate)

    if gpu:
        auto_encoder.cuda()

    # Initialize variables
    best_loss_valid = np.inf
    print("Beginning training")
    for epoch in range(options.transfer_learning_epochs):
        print("At %d-th epoch." % epoch)

        auto_encoder.zero_grad()
        evaluation_flag = True
        step_flag = True
        last_check_point_i = 0
        for i, data in enumerate(train_loader):
            if gpu:
                imgs = data['image'].cuda()
            else:
                imgs = data['image']

            hidden = first_layers(imgs)
            train_output = auto_encoder(hidden)
            loss = loss_func(train_output, hidden)
            loss.backward()

            # writer_train.add_scalar('training_loss', loss.item() / len(data), i + epoch * len(train_loader.dataset))

            if (i+1) % options.accumulation_steps == 0:
                step_flag = False
                optimizer.step()
                optimizer.zero_grad()

                # Evaluate the decoder only when no gradients are accumulated
                if (i+1) % options.evaluation_steps == 0:
                    evaluation_flag = False
                    print('Iteration %d' % i)
                    loss_train = test_ae(auto_encoder, train_loader, gpu, loss_func, first_layers=first_layers)
                    mean_loss_train = loss_train / (len(train_loader) * train_loader.dataset.size)
                    loss_valid = test_ae(auto_encoder, valid_loader, gpu, loss_func, first_layers=first_layers)
                    mean_loss_valid = loss_valid / (len(valid_loader) * valid_loader.dataset.size)
                    auto_encoder.train()
                    print("Scan level validation loss is %f at the end of iteration %d" % (loss_valid, i))

                    row = np.array([epoch, i, loss_train, mean_loss_train, loss_valid, mean_loss_valid]).reshape(1, -1)
                    row_df = pd.DataFrame(row, columns=columns)
                    with open(filename, 'a') as f:
                        row_df.to_csv(f, header=False, index=False, sep='\t')

            del imgs

        # If no step has been performed, raise Exception
        if step_flag:
            raise Exception('The model has not been updated once in the epoch. The accumulation step may be too large.')

        # If no evaluation has been performed, warn the user
        if evaluation_flag:
            warnings.warn('Your evaluation steps are too big compared to the size of the dataset.'
                          'The model is evaluated only once at the end of the epoch')

        # Always test the results and save them once at the end of the epoch
        if last_check_point_i != i:
            print('Last checkpoint at the end of the epoch %d' % epoch)
            loss_train = test_ae(auto_encoder, train_loader, gpu, loss_func, first_layers=first_layers)
            mean_loss_train = loss_train / (len(train_loader) * train_loader.dataset.size)
            loss_valid = test_ae(auto_encoder, valid_loader, gpu, loss_func, first_layers=first_layers)
            mean_loss_valid = loss_valid / (len(valid_loader) * valid_loader.dataset.size)
            auto_encoder.train()
            print("Scan level validation loss is %f at the end of iteration %d" % (loss_valid, i))

            row = np.array([epoch, i, loss_train, mean_loss_train, loss_valid, mean_loss_valid]).reshape(1, -1)
            row_df = pd.DataFrame(row, columns=columns)
            with open(filename, 'a') as f:
                row_df.to_csv(f, header=False, index=False, sep='\t')

            is_best = loss_valid < best_loss_valid
            # Save only if is best to avoid performance deterioration
            if is_best:
                best_loss_valid = loss_valid
                save_checkpoint({'model': auto_encoder.state_dict(),
                                 'iteration': i,
                                 'epoch': epoch,
                                 'loss_valid': loss_valid},
                                is_best,
                                results_path)


def extract_ae(decoder, level):
    import torch.nn as nn
    from model import Decoder

    n_conv = 0
    output_decoder = Decoder()
    inverse_layers = []

    for i, layer in enumerate(decoder.encoder):
        if isinstance(layer, nn.Conv3d):
            n_conv += 1

        if n_conv == level + 1:
            output_decoder.encoder.add_module(str(len(output_decoder.encoder)), layer)
            # Do not keep two successive BatchNorm layers
            if not isinstance(layer, nn.BatchNorm3d):
                inverse_layers.append(decoder.decoder[len(decoder.decoder) - (i + 1)])

        elif n_conv > level + 1:
            break

    inverse_layers.reverse()
    output_decoder.decoder = nn.Sequential(*inverse_layers)
    return output_decoder


def extract_first_layers(decoder, level):
    import torch.nn as nn
    from copy import deepcopy
    from modules import PadMaxPool3d

    n_conv = 0
    first_layers = nn.Sequential()

    for i, layer in enumerate(decoder.encoder):
        if isinstance(layer, nn.Conv3d):
            n_conv += 1

        if n_conv < level + 1:
            layer_copy = deepcopy(layer)
            layer_copy.requires_grad = False
            if isinstance(layer, PadMaxPool3d):
                layer_copy.set_new_return(False, False)

            first_layers.add_module(str(i), layer_copy)
        else:
            break

    return first_layers


def visualize_ae(decoder, dataloader, results_path, gpu):
    import nibabel as nib
    from data_utils import ToTensor
    import os
    from os import path

    if not path.exists(results_path):
        os.makedirs(results_path)

    subject = dataloader.dataset.df.loc[0, 'participant_id']
    session = dataloader.dataset.df.loc[0, 'session_id']

    img_path = path.join(dataloader.dataset.img_dir, 'subjects', subject, session,
                         't1', 'preprocessing_dl',
                         subject + '_' + session + '_space-MNI_res-1x1x1_linear_registration.nii.gz')
    data = nib.load(img_path)
    img = data.get_data()
    affine = data.get_affine()
    img_tensor = ToTensor()(img)
    img_tensor = img_tensor.unsqueeze(0)
    if gpu:
        img_tensor = img_tensor.cuda()
    print(img_tensor.size())
    output_tensor = decoder(img_tensor)
    output = nib.Nifti1Image(output_tensor[0].cpu().detach().numpy(), affine)
    nib.save(output, os.path.join(results_path, 'output_image.nii'))
    nib.save(data, os.path.join(results_path, 'input_image.nii'))


def memReport():
    import gc

    cnt_tensor = 0
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            print(type(obj), obj.size(), obj.is_cuda)
            cnt_tensor += 1
    print('Count: ', cnt_tensor)


def cpuStats():
    import sys
    import psutil

    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', memoryUse)

def results_to_tsvs(output_dir, iteration, subject_list, y_truth, y_hat, mode='train'):
    """
    This is a function to trace all subject during training, test and validation, and calculate the performances with different metrics into tsv files.
    :param output_dir:
    :param iteration:
    :param subject_list:
    :param y_truth:
    :param y_hat:
    :return:
    """

    # check if the folder exist
    iteration_dir = os.path.join(output_dir, 'performances', 'iteration-' + str(iteration))
    if not os.path.exists(iteration_dir):
        os.makedirs(iteration_dir)
    iteration_subjects_df = pd.DataFrame({'iteration': iteration,
                                                'y': y_truth,
                                                'y_hat': y_hat,
                                                'subject': subject_list})
    iteration_subjects_df.to_csv(os.path.join(iteration_dir, mode + '_subjects.tsv'), index=False, sep='\t', encoding='utf-8')

    results = evaluate_prediction(np.asarray(y_truth), np.asarray(y_hat))
    del results['confusion_matrix']
    pd.DataFrame(results, index=[0]).to_csv(os.path.join(iteration_dir, mode + '_result.tsv'), index=False, sep='\t', encoding='utf-8')

    return iteration_subjects_df, pd.DataFrame(results, index=[0])

def evaluate_prediction(y, y_hat):

    true_positive = 0.0
    true_negative = 0.0
    false_positive = 0.0
    false_negative = 0.0

    tp = []
    tn = []
    fp = []
    fn = []

    for i in range(len(y)):
        if y[i] == 1:
            if y_hat[i] == 1:
                true_positive += 1
                tp.append(i)
            else:
                false_negative += 1
                fn.append(i)
        else:  # -1
            if y_hat[i] == 0:
                true_negative += 1
                tn.append(i)
            else:
                false_positive += 1
                fp.append(i)

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
               'confusion_matrix': {'tp': len(tp), 'tn': len(tn), 'fp': len(fp), 'fn': len(fn)}
               }

    return results