import numpy as np
import os
import warnings

from clinicadl.tools.deep_learning.iotools import check_and_clean
from clinicadl.tools.deep_learning import EarlyStopping, save_checkpoint


#############################
# AutoEncoder train / test  #
#############################

def train(decoder, train_loader, valid_loader, criterion, optimizer, resume,
          log_dir, model_dir, visualization_dir, options):
    """
    Function used to train an autoencoder.
    The best autoencoder and checkpoint will be found in the 'best_model_dir' of options.output_dir.

    :param decoder: (Autoencoder) Autoencoder constructed from a CNN with the Autoencoder class.
    :param train_loader: (DataLoader) wrapper of the training dataset.
    :param valid_loader: (DataLoader) wrapper of the validation dataset.
    :param criterion: (loss) function to calculate the loss.
    :param optimizer: (torch.optim) optimizer linked to model parameters.
    :param resume: (bool) if True, a begun job is resumed.
    :param log_dir: (str) path to the folder containing the logs.
    :param model_dir: (str) path to the folder containing the models weights and biases.
    :param visualization_dir: (str) path to the folder containing the reconstruction of images.
    :param options: (Namespace) ensemble of other options given to the main script.
    """
    from tensorboardX import SummaryWriter

    if not resume:
        check_and_clean(model_dir)
        check_and_clean(visualization_dir)
        check_and_clean(log_dir)
        options.beginning_epoch = 0

    # Create writers
    writer_train = SummaryWriter(os.path.join(log_dir, 'train'))
    writer_valid = SummaryWriter(os.path.join(log_dir, 'valid'))

    decoder.train()
    print(decoder)

    if options.gpu:
        decoder.cuda()

    # Initialize variables
    best_loss_valid = np.inf

    early_stopping = EarlyStopping('min', min_delta=options.tolerance, patience=options.patience)
    loss_valid = None
    epoch = options.beginning_epoch

    print("Beginning training")
    while epoch < options.epochs and not early_stopping.step(loss_valid):
        print("At %d-th epoch." % epoch)

        decoder.zero_grad()
        evaluation_flag = True
        step_flag = True
        for i, data in enumerate(train_loader):
            if options.gpu:
                imgs = data['image'].cuda()
            else:
                imgs = data['image']

            train_output = decoder(imgs)
            loss = criterion(train_output, imgs)
            loss.backward()

            del imgs, train_output

            if (i+1) % options.accumulation_steps == 0:
                step_flag = False
                optimizer.step()
                optimizer.zero_grad()

                # Evaluate the decoder only when no gradients are accumulated
                if options.evaluation_steps != 0 and (i + 1) % options.evaluation_steps == 0:
                    evaluation_flag = False
                    print('Iteration %d' % i)
                    loss_train = test_ae(decoder, train_loader, options.gpu, criterion)
                    mean_loss_train = loss_train / (len(train_loader) * train_loader.batch_size)

                    loss_valid = test_ae(decoder, valid_loader, options.gpu, criterion)
                    mean_loss_valid = loss_valid / (len(valid_loader) * valid_loader.batch_size)
                    decoder.train()

                    writer_train.add_scalar('loss', mean_loss_train, i + epoch * len(train_loader))
                    writer_valid.add_scalar('loss', mean_loss_valid, i + epoch * len(train_loader))
                    print("Scan level validation loss is %f at the end of iteration %d" % (loss_valid, i))

        # If no step has been performed, raise Exception
        if step_flag:
            raise Exception('The model has not been updated once in the epoch. The accumulation step may be too large.')

        # If no evaluation has been performed, warn the user
        if evaluation_flag and options.evaluation_steps != 0:
            warnings.warn('Your evaluation steps are too big compared to the size of the dataset.'
                          'The model is evaluated only once at the end of the epoch')

        # Always test the results and save them once at the end of the epoch
        print('Last checkpoint at the end of the epoch %d' % epoch)

        loss_train = test_ae(decoder, train_loader, options.gpu, criterion)
        mean_loss_train = loss_train / (len(train_loader) * train_loader.batch_size)

        loss_valid = test_ae(decoder, valid_loader, options.gpu, criterion)
        mean_loss_valid = loss_valid / (len(valid_loader) * valid_loader.batch_size)
        decoder.train()

        writer_train.add_scalar('loss', mean_loss_train, i + epoch * len(train_loader))
        writer_valid.add_scalar('loss', mean_loss_valid, i + epoch * len(train_loader))
        print("Scan level validation loss is %f at the end of iteration %d" % (loss_valid, i))

        is_best = loss_valid < best_loss_valid
        best_loss_valid = min(best_loss_valid, loss_valid)
        # Always save the model at the end of the epoch and update best model
        save_checkpoint({'model': decoder.state_dict(),
                         'epoch': epoch,
                         'valid_loss': loss_valid},
                        False, is_best,
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


def test_ae(model, dataloader, use_cuda, criterion):
    """
    Computes the loss of the model

    :param model: the network (subclass of nn.Module)
    :param dataloader: a DataLoader wrapping a dataset
    :param use_cuda: if True a gpu is used
    :param criterion: (loss) function to calculate the loss
    :return: loss of the model (float)
    """
    model.eval()

    total_loss = 0
    for i, data in enumerate(dataloader, 0):
        if use_cuda:
            inputs = data['image'].cuda()
        else:
            inputs = data['image']

        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        total_loss += loss.item()

        del inputs, outputs, loss

    return total_loss


def visualize_subject(decoder, dataloader, visualization_path, options, epoch=None, save_input=True, subject_index=0):
    from os import path, makedirs, pardir
    import nibabel as nib
    import numpy as np
    import torch
    from .data import MinMaxNormalization

    if not path.exists(visualization_path):
        makedirs(visualization_path)

    dataset = dataloader.dataset
    data = dataset[subject_index]
    image_path = data['image_path']

    # TODO: Change nifti path
    nii_path, _ = path.splitext(image_path)
    nii_path += '.nii.gz'

    if not path.exists(nii_path):
        nii_path = path.join(
            path.dirname(image_path),
            pardir, pardir, pardir,
            't1_linear',
            path.basename(image_path)
        )
        nii_path, _ = path.splitext(nii_path)
        nii_path += '.nii.gz'

    input_nii = nib.load(nii_path)
    input_np = input_nii.get_data().astype(float)
    np.nan_to_num(input_np, copy=False)
    input_pt = torch.from_numpy(input_np).unsqueeze(0).unsqueeze(0).float()
    if options.minmaxnormalization:
        transform = MinMaxNormalization()
        input_pt = transform(input_pt)

    if options.gpu:
        input_pt = input_pt.cuda()

    output_pt = decoder(input_pt)

    output_np = output_pt.detach().cpu().numpy()[0][0]
    output_nii = nib.Nifti1Image(output_np, affine=input_nii.affine)

    if save_input:
        nib.save(input_nii, path.join(visualization_path, 'input.nii'))

    if epoch is None:
        nib.save(output_nii, path.join(visualization_path, 'output.nii'))
    else:
        nib.save(output_nii, path.join(visualization_path, 'epoch-' + str(epoch) + '.nii'))