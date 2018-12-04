import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import shutil
import warnings
import pandas as pd


def train(model, train_loader, valid_loader, criterion, optimizer, fold, options):
    """
    This is the function to train the model
    :param model:
    :param train_loader:
    :param valid_loader:
    :param criterion:
    :param optimizer:
    :param fold:
    :param options:
    """
    # Create writers
    from tensorboardX import SummaryWriter
    from time import time

    writer_train = SummaryWriter(log_dir=(os.path.join(options.log_dir, "fold" + str(fold), "train")))  # Replace with a path creation
    filename = os.path.join(options.log_dir, "fold" + str(fold), 'training.tsv')
    results_df = pd.DataFrame(columns=['epoch', 'iteration', 'acc_train', 'acc_valid'])
    with open(filename, 'w') as f:
        results_df.to_csv(f, index=False, sep='\t')

    # Initialize variables
    best_valid_accuracy = 0.0
    total_acc = 0.0
    epoch = 0

    model.train()  # set the module to training mode

    while epoch < options.epochs and total_acc < 1 - options.tolerance:
        total_correct_cnt = 0.0
        print("At %d-th epoch." % epoch)

        model.zero_grad()
        evaluation_flag = True
        step_flag = True
        last_check_point_i = 0
        for i, data in enumerate(train_loader):

            if options.gpu:
                imgs, labels = data['image'].cuda(), data['label'].cuda()
            else:
                imgs, labels = data['image'], data['label']

            train_output = model(imgs)
            _, predict = train_output.topk(1)
            loss = criterion(train_output, labels)
            batch_correct_cnt = (predict.squeeze(1) == labels).sum().float()
            total_correct_cnt += batch_correct_cnt
            # accuracy = float(batch_correct_cnt) / len(labels)
            loss.backward()

            # writer_train.add_scalar('training_accuracy', accuracy / len(data), i + epoch * len(train_loader.dataset))
            # writer_train.add_scalar('training_loss', loss.item() / len(data), i + epoch * len(train_loader.dataset))

            del imgs

            if (i+1) % options.accumulation_steps == 0:
                step_flag = False
                optimizer.step()
                model.zero_grad()

                # Evaluate the model only when no gradients are accumulated
                if(i+1) % options.evaluation_steps == 0:
                    evaluation_flag = False
                    print('Iteration %d' % i)
                    acc_mean_train = test(model, train_loader, options.gpu)
                    acc_mean_valid = test(model, valid_loader, options.gpu)
                    model.train()
                    print("Scan level training accuracy is %f at the end of iteration %d" % (acc_mean_train, i))
                    print("Scan level validation accuracy is %f at the end of iteration %d" % (acc_mean_valid, i))

                    row = np.array([epoch, i, acc_mean_train, acc_mean_valid]).reshape(1, -1)
                    row_df = pd.DataFrame(row, columns=['epoch', 'iteration', 'acc_train', 'acc_valid'])
                    with open(filename, 'a') as f:
                        row_df.to_csv(f, header=False, index=False, sep='\t')
                    is_best = acc_mean_valid > best_valid_accuracy
                    # Save only if is best to avoid performance deterioration
                    if is_best:
                        best_valid_accuracy = acc_mean_valid
                        save_checkpoint({'model': model.state_dict(),
                                         'iteration': i,
                                         'epoch': epoch,
                                         'valid_acc': acc_mean_valid},
                                        is_best,
                                        os.path.join(options.log_dir, "fold" + str(fold)))
                        last_check_point_i = i

            if (i + 1) % 10 == 0:
                print('Batch: ' + str(i))


        # If no step has been performed, raise Exception
        if step_flag:
            raise Exception('The model has not been updated once in the epoch. The accumulation step may be too large.')

        # If no evaluation has been performed, warn the user
        elif evaluation_flag:
            warnings.warn('Your evaluation steps are too big compared to the size of the dataset.'
                          'The model is evaluated only once at the end of the epoch')

        # Always test the results and save them once at the end of the epoch
        if last_check_point_i != i:
            model.zero_grad()
            print('Last checkpoint at the end of the epoch %d' % epoch)
            acc_mean_train = test(model, train_loader, options.gpu)
            acc_mean_valid = test(model, valid_loader, options.gpu)
            model.train()
            print("Scan level training accuracy is %f at the end of iteration %d" % (acc_mean_train, i))
            print("Scan level validation accuracy is %f at the end of iteration %d" % (acc_mean_valid, i))

            row = np.array([epoch, i, acc_mean_train, acc_mean_valid]).reshape(1, -1)
            row_df = pd.DataFrame(row, columns=['epoch', 'iteration', 'acc_train', 'acc_valid'])
            with open(filename, 'a') as f:
                row_df.to_csv(f, header=False, index=False, sep='\t')
            is_best = acc_mean_valid > best_valid_accuracy
            best_valid_accuracy = max(acc_mean_valid, best_valid_accuracy)
            save_checkpoint({'model': model.state_dict(),
                             'epoch': epoch,
                             'valid_acc': acc_mean_valid},
                            is_best,
                            os.path.join(options.log_dir, "fold" + str(fold)))

        print('Total correct labels: %d / %d' % (total_correct_cnt, len(train_loader) * train_loader.batch_size))
        total_acc = float(total_correct_cnt) / (len(train_loader) * train_loader.batch_size)
        epoch += 1


def test(model, dataloader, use_cuda, verbose=False, full_return=False):
    """
    Computes the balanced accuracy of the model

    :param model: the network (subclass of nn.Module)
    :param dataloader: a DataLoader wrapping a dataset
    :param use_cuda: if True a gpu is used
    :param full_return: if True also returns the sensitivities and specificities for a multiclass problem
    :return: balanced accuracy of the model (float)
    """
    model.eval()

    # Use tensors instead of arrays to avoid bottlenecks
    predicted_tensor = torch.zeros(len(dataloader.dataset))
    truth_tensor = torch.zeros(len(dataloader.dataset))
    if use_cuda:
        predicted_tensor = predicted_tensor.cuda()
        truth_tensor = truth_tensor.cuda()

    for i, data in enumerate(dataloader):
        if use_cuda:
            inputs, labels = data['image'].cuda(), data['label'].cuda()
        else:
            inputs, labels = data['image'], data['label']

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        idx = i * dataloader.batch_size
        idx_end = (i + 1) * dataloader.batch_size
        predicted_tensor[idx:idx_end:] = predicted
        truth_tensor[idx:idx_end:] = labels

        del inputs, outputs, labels

    print('After testing', torch.cuda.memory_allocated())
    # Computation of the balanced accuracy
    component = len(np.unique(truth_tensor))

    # Cast to numpy arrays to avoid bottleneck in the next loop
    if use_cuda:
        predicted_arr = predicted_tensor.cpu().numpy().astype(int)
        truth_arr = truth_tensor.cpu().numpy().astype(int)
    else:
        predicted_arr = predicted_tensor.numpy()
        truth_arr = truth_tensor.numpy()

    cluster_diagnosis_prop = np.zeros(shape=(component, component))
    for i, predicted in enumerate(predicted_arr):
        truth = truth_arr[i]
        cluster_diagnosis_prop[predicted, truth] += 1

    print("Processing data", torch.cuda.memory_allocated())

    acc = 0
    sensitivity = np.zeros(component)
    specificity = np.zeros(component)
    for i in range(component):
        diag_represented = np.argmax(cluster_diagnosis_prop[i])
        acc += cluster_diagnosis_prop[i, diag_represented] / np.sum(cluster_diagnosis_prop.T[diag_represented])

        # Computation of sensitivity
        sen_array = cluster_diagnosis_prop[i]
        if np.sum(sen_array) == 0:
            sensitivity[diag_represented] = None
        else:
            sensitivity[diag_represented] = sen_array[diag_represented] / np.sum(sen_array) * 100

        # Computation of specificity
        spe_array = np.delete(cluster_diagnosis_prop, i, 0)
        if np.sum(spe_array) == 0:
            specificity[diag_represented] = None
        else:
            specificity[diag_represented] = (1 - np.sum(spe_array[:, diag_represented]) / np.sum(spe_array)) * 100

    print('Postprocessing', torch.cuda.memory_allocated())
    acc = acc * 100 / component
    if verbose:
        print('Accuracy of diagnosis: ' + str(acc))
        print('Sensitivity of diagnoses:', sensitivity)
        print('Specificity of diagnoses:', specificity)

    if full_return:
        return acc, sensitivity, specificity

    return acc


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


def ae_pretraining(model, train_loader, valid_loader, criterion, gpu, options):
    from model import Decoder
    from tensorboardX import SummaryWriter
    from copy import deepcopy

    writer_train = SummaryWriter(log_dir=(os.path.join(options.log_dir, "pretraining")))

    decoder = Decoder(model)
    decoder.train()
    optimizer = eval("torch.optim." + options.optimizer)(filter(lambda x: x.requires_grad, decoder.parameters()),
                                                         options.learning_rate)

    if gpu:
        decoder.cuda()

    # Initialize variables
    best_loss_valid = np.inf
    print("Beginning pretraining")
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
            loss = criterion(train_output, imgs)
            loss.backward()

            writer_train.add_scalar('training_loss', loss.item() / len(data), i + epoch * len(train_loader.dataset))

            if (i+1) % options.accumulation_steps == 0:
                step_flag = False
                optimizer.step()
                model.zero_grad()

                # Evaluate the decoder only when no gradients are accumulated
                if (i + 1) % options.evaluation_steps == 0:
                    evaluation_flag = False
                    print('Iteration %d' % i)
                    loss_valid = test_ae(decoder, valid_loader, gpu, criterion)
                    decoder.train()
                    print("Scan level validation loss is %f at the end of iteration %d" % (loss_valid, i))

                    is_best = loss_valid < best_loss_valid
                    # Save only if is best to avoid performance deterioration
                    if is_best:
                        best_loss_valid = loss_valid
                        save_checkpoint({'model': decoder.state_dict(),
                                         'iteration': i,
                                         'epoch': epoch,
                                         'loss_valid': loss_valid},
                                        is_best,
                                        os.path.join(options.log_dir, "pretraining"))
                        last_check_point_i = i

                if (i+1) % 2 == 0:
                    print('Batch: ' + str(i))

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
            loss_valid = test_ae(decoder, valid_loader, gpu, criterion)
            decoder.train()
            print("Scan level validation loss is %f at the end of iteration %d" % (loss_valid, i))

            is_best = loss_valid < best_loss_valid
            # Save only if is best to avoid performance deterioration
            if is_best:
                best_loss_valid = loss_valid
                save_checkpoint({'model': decoder.state_dict(),
                                 'iteration': i,
                                 'epoch': epoch,
                                 'loss_valid': loss_valid},
                                is_best,
                                os.path.join(options.log_dir, "pretraining"))

    # print('End of training', torch.cuda.memory_allocated())
    # Updating and setting weights of the convolutional layers
    best_decoder, best_epoch = load_model(decoder, os.path.join(options.log_dir, "pretraining"))
    model.features = deepcopy(best_decoder.encoder)
    save_checkpoint({'model': model.state_dict(),
                     'epoch': best_epoch},
                    False,
                    os.path.join(options.log_dir, "pretraining"),
                    'model_pretrained.pth.tar')

    if options.visualization is not None:
        visualize_ae(best_decoder, options.visualization, os.path.join(options.log_dir, "pretraining"), gpu)


def test_ae(model, dataloader, use_cuda, criterion):
    """
    Computes the loss of the model

    :param model: the network (subclass of nn.Module)
    :param dataloader: a DataLoader wrapping a dataset
    :param use_cuda: if True a gpu is used
    :return: loss of the model (float)
    """
    model.eval()

    total_loss = 0
    for i, data in enumerate(dataloader):
        if use_cuda:
            inputs = data['image'].cuda()
        else:
            inputs = data['image']

        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        total_loss += loss.item()

    return total_loss


def visualize_ae(decoder, img_path, results_path, gpu):
    import nibabel as nib
    from data_utils import ToTensor
    import os

    data = nib.load(img_path)
    img = data.get_data()
    affine = data.get_affine()
    img_tensor = ToTensor()(img)
    img_tensor = img_tensor.unsqueeze(0)
    if gpu:
        img_tensor = img_tensor.cuda()
    print(img_tensor.size())
    output_tensor = decoder(img_tensor)
    output = nib.Nifti1Image(output_tensor.cpu().detach().numpy(), affine)
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