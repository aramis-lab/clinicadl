import numpy as np
import os
import warnings

from clinicadl.tools.deep_learning.iotools import check_and_clean
from clinicadl.tools.deep_learning import EarlyStopping, save_checkpoint


#############################
# AutoEncoder train / test  #
#############################

def train(decoder, train_loader, valid_loader, criterion, optimizer, resume,
          log_dir, model_dir, options):
    """
    Function used to train an autoencoder.
    The best autoencoder will be found in the 'best_model_dir' of options.output_dir.

    Args:
        decoder: (Autoencoder) Autoencoder constructed from a CNN with the Autoencoder class.
        train_loader: (DataLoader) wrapper of the training dataset.
        valid_loader: (DataLoader) wrapper of the validation dataset.
        criterion: (loss) function to calculate the loss.
        optimizer: (torch.optim) optimizer linked to model parameters.
        resume: (bool) if True, a begun job is resumed.
        log_dir: (str) path to the folder containing the logs.
        model_dir: (str) path to the folder containing the models weights and biases.
        options: (Namespace) ensemble of other options given to the main script.
    """
    from tensorboardX import SummaryWriter

    if not resume:
        check_and_clean(model_dir)
        check_and_clean(log_dir)
        options.beginning_epoch = 0

    # Create writers
    writer_train = SummaryWriter(os.path.join(log_dir, 'train'))
    writer_valid = SummaryWriter(os.path.join(log_dir, 'validation'))

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

            if (i + 1) % options.accumulation_steps == 0:
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


def test_ae(decoder, dataloader, use_cuda, criterion):
    """
    Computes the total loss of a given autoencoder and dataset wrapped by DataLoader.

    Args:
        decoder: (Autoencoder) Autoencoder constructed from a CNN with the Autoencoder class.
        dataloader: (DataLoader) wrapper of the dataset.
        use_cuda: (bool) if True a gpu is used.
        criterion: (loss) function to calculate the loss.

    Returns:
        (float) total loss of the model
    """
    decoder.eval()

    total_loss = 0
    for i, data in enumerate(dataloader, 0):
        if use_cuda:
            inputs = data['image'].cuda()
        else:
            inputs = data['image']

        outputs = decoder(inputs)
        loss = criterion(outputs, inputs)
        total_loss += loss.item()

        del inputs, outputs, loss

    return total_loss


def visualize_image(decoder, dataloader, visualization_path, nb_images=1):
    """
    Writes the nifti files of images and their reconstructions by an autoencoder.

    Args:
        decoder: (Autoencoder) Autoencoder constructed from a CNN with the Autoencoder class.
        dataloader: (DataLoader) wrapper of the dataset.
        visualization_path: (str) directory in which the inputs and reconstructions will be stored.
        nb_images: (int) number of images to reconstruct.
    """
    import nibabel as nib
    import numpy as np
    from .iotools import check_and_clean

    check_and_clean(visualization_path)

    dataset = dataloader.dataset
    decoder.eval()

    for image_index in range(nb_images):
        data = dataset[image_index]
        image = data["image"].unsqueeze(0)
        output = decoder(image)

        output_np = output.squeeze(0).squeeze(0).cpu().detach().numpy()
        input_np = image.squeeze(0).squeeze(0).cpu().detach().numpy()
        output_nii = nib.Nifti1Image(output_np, np.eye(4))
        input_nii = nib.Nifti1Image(input_np, np.eye(4))
        nib.save(output_nii, os.path.join(visualization_path, 'output-%i.nii.gz' % image_index))
        nib.save(input_nii, os.path.join(visualization_path, 'input-%i.nii.gz' % image_index))
