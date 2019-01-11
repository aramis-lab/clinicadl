import argparse
import nibabel as nib

from classification_utils import *
from data_utils import *
from model import *

parser = argparse.ArgumentParser(description="Argparser for Pytorch 3D AE pretraining")

# Mandatory arguments
parser.add_argument("diagnosis_path", type=str,
                    help="Path to the folder containing the tsv files of the population.")
parser.add_argument("model_path", type=str,
                    help="Path to the trained model folder.")
parser.add_argument("input_dir", type=str,
                    help="Path to input dir of the MRI (preprocessed CAPS_dir).")
parser.add_argument("model", type=str,
                    help="model selected")

# Data Management
parser.add_argument("--diagnoses", "-d", default=["AD", "CN"], nargs='+', type=str,
                    help="Take all the subjects possible for autoencoder training")
parser.add_argument("--gpu", action="store_true", default=False,
                    help="if True computes the visualization on GPU")
parser.add_argument("--minmaxnormalization", "-n", default=False, action="store_true",
                    help="Performs MinMaxNormalization for visualization")
parser.add_argument("--feature_maps", "-fm", default=False, action="store_true",
                    help="Performs feature maps extraction and visualization")
parser.add_argument("--filters", "-f", default=False, action="store_true",
                    help="Performs the visualization of filters by optimizing images which maximally activate"
                         "each filter.")


def feature_maps_extraction(decoder, input_pt, results_path, affine, level=0):
    """
    Visualization of the feature maps of different levels (from 0 to decoder.level - 1)

    :param decoder: (Decoder)
    :param input_pt: (tensor) the input image
    :param results_path: (str) path to the visualization
    :param affine: (np array) Nifti parameter to construct images
    :param level: (int) the number of convolutional levels kept to extract the feature maps
    :return:
        None
    """
    from copy import copy

    affine_level = copy(affine)
    affine_level[0:3, 0:3] = affine_level[0:3, 0:3] * 2**(level + 1)
    if level >= decoder.level:
        raise ValueError("The feature maps level %s cannot be superior or equal to the decoder level %s."
                         % (level, decoder.level))

    encoder = extract_first_layers(decoder, level + 1)
    output_pt = encoder(input_pt)
    n_feature_maps = output_pt.size(1)

    for i in range(n_feature_maps):
        feature_map_np = output_pt.detach().numpy()[0][i]
        feature_map_nii = nib.Nifti1Image(feature_map_np, affine=affine_level)

        nib.save(feature_map_nii, path.join(results_path, 'level-' + str(level) + '_feature_map-' + str(i)))


def weight_visualization(decoder, results_path, affine, level=0, previous_level=None):

    coeff_level = 2 ** (level + 1)

    if previous_level is None:
        first_layer = decoder.encoder[level * int(len(decoder) / decoder.level)].weight
    else:
        first_layer = previous_level

    second_layer = decoder.encoder[(level + 1) * int(len(decoder) / decoder.level)].weight

    print('Shape of the first layer', first_layer.shape)
    print('Shape of the second layer', second_layer.shape)

    n_filters_second = second_layer.size(0)
    n_filters_first = first_layer.size(0)
    channels = first_layer.size(1)

    output = torch.zeros(n_filters_second, channels,
                         first_layer.size(2) * 2 + 1,
                         first_layer.size(3) * 2 + 1,
                         first_layer.size(4) * 2 + 1)

    for i in range(n_filters_second):
        for j in range(n_filters_first):
            filter_first_layer = first_layer[j]
            for x in range(3):
                for y in range(3):
                    for z in range(3):
                        new_values = filter_first_layer * second_layer[i, j, x, y, z]
                        output[i, :,
                               coeff_level * x: coeff_level * x + first_layer.size(2),
                               coeff_level * y: coeff_level * y + first_layer.size(3),
                               coeff_level * z: coeff_level * z + first_layer.size(4)] = new_values

        output_filter_np = output[i][0].detach().numpy()
        output_filter_nii = nib.Nifti1Image(output_filter_np, affine)
        nib.save(output_filter_nii, path.join(results_path, 'level-' + str(level) + '_filter-' + str(i)))

    return output


class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter, size):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        # Generate a random image
        self.created_image = np.random.uniform(0.0, 1.0, size)

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]

        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self, results_path, affine):
        # Create path
        if not path.join(results_path, 'layer_vis'):
            os.makedirs(path.join(results_path, 'layer_vis'))
        # Hook the selected layer
        self.hook_layer()
        # Process image and return variable
        transform = ToTensor()
        processed_image = transform(self.created_image)
        processed_image = processed_image.unsqueeze(0)
        # Define optimizer for the image
        optimizer = torch.optim.Adam([processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, 101):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image
            for index, layer in enumerate(self.model.encoder):
                # Forward pass layer by layer
                # x is not used after this point because it is only needed to trigger
                # the forward hook function
                if isinstance(layer, PadMaxPool3d):
                    x, indices, pad = layer(x)
                elif isinstance(layer, nn.MaxPool3d):
                    x, indices = layer(x)
                else:
                    x = layer(x)

                # Only need to forward until the selected layer is reached
                if index == self.selected_layer:
                    # (forward hook function triggered)
                    break
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = processed_image.detach().numpy()[0][0]
            # Save image
            # if i % 20 == 0:
        im_path = path.join(results_path, 'layer_vis', 'layer-' + str(self.selected_layer) +
                            '_f' + str(self.selected_filter) + '_iter' + str(i) + '.nii')
        create_image_nii = nib.Nifti1Image(self.created_image, affine)
        nib.save(create_image_nii, im_path)

    def visualise_layer_without_hooks(self, results_path, affine):
        # Create path
        if not path.exists(path.join(results_path, 'layer_vis')):
            os.makedirs(path.join(results_path, 'layer_vis'))
        # Process image and return variable
        transform = ToTensor()
        processed_image = transform(self.created_image)
        processed_image = processed_image.unsqueeze(0)
        processed_image.requires_grad = True
        # Define optimizer for the image
        optimizer = torch.optim.Adam([processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, 101):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image
            for index, layer in enumerate(self.model.encoder):
                # Forward pass layer by layer
                # x is not used after this point because it is only needed to trigger
                # the forward hook function
                if isinstance(layer, PadMaxPool3d):
                    x, indices, pad = layer(x)
                elif isinstance(layer, nn.MaxPool3d):
                    x, indices = layer(x)
                else:
                    x = layer(x)

                if index == self.selected_layer:
                    # Only need to forward until the selected layer is reached
                    # Now, x is the output of the selected layer
                    break
            # Here, we get the specific filter from the output of the convolution operation
            # x is a tensor of shape 1x512x28x28.(For layer 17)
            # So there are 512 unique filter outputs
            # Following line selects a filter from 512 filters so self.conv_output will become
            # a tensor of shape 28x28
            self.conv_output = x[0, self.selected_filter]
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = processed_image.detach().numpy()[0][0]
            # Save image

        im_path = path.join(results_path, 'layer_vis', 'layer-' + str(self.selected_layer) +
                            '_f' + str(self.selected_filter) + '_iter' + str(i) + '.nii')
        create_image_nii = nib.Nifti1Image(self.created_image, affine)
        nib.save(create_image_nii, im_path)


def main(options):

    # Check if model is implemented
    import model
    import inspect

    choices = []
    for name, obj in inspect.getmembers(model):
        if inspect.isclass(obj):
            choices.append(name)

    if options.model not in choices:
        raise NotImplementedError('The model wanted %s has not been implemented in the module model.py' % options.model)

    model = eval(options.model)()
    decoder = Decoder(model)
    best_decoder, _ = load_model(decoder, options.model_path)
    sets = ['train', 'validation']

    for set in sets:

        if options.minmaxnormalization:
            set_path = path.join(options.model_path, "visualization_norm", set)
        else:
            set_path = path.join(options.model_path, "visualization", set)

        if not path.exists(set_path):
            os.makedirs(set_path)

        for diagnosis in options.diagnoses:

            diagnosis_path = path.join(set_path, diagnosis)
            if not path.exists(diagnosis_path):
                os.makedirs(diagnosis_path)

            set_df = pd.read_csv(path.join(options.diagnosis_path, set, diagnosis + '_baseline.tsv'), sep='\t')
            subject = set_df.loc[0, 'participant_id']
            session = set_df.loc[0, 'session_id']
            image_path = path.join(options.input_dir, 'subjects', subject, session,
                                   't1', 'preprocessing_dl',
                                   subject + '_' + session + '_space-MNI_res-1x1x1.nii.gz')
            input_nii = nib.load(image_path)
            input_np = input_nii.get_data()
            input_pt = torch.from_numpy(input_np).unsqueeze(0).unsqueeze(0).float()
            if options.minmaxnormalization:
                transform = MinMaxNormalization()
                input_pt = transform(input_pt)

            if options.feature_maps:

                for level in range(decoder.level):
                    feature_maps_extraction(decoder, input_pt, diagnosis_path, input_nii.affine, level)

                    selected_layer = level * int(len(decoder) / decoder.level)

            if options.filters:

                for level in range(decoder.level):
                    n_filters = decoder.encoder[selected_layer].weight.size(0)
                    for selected_filter in range(n_filters):
                        visualization_tool = CNNLayerVisualization(decoder, selected_layer, selected_filter,
                                                                   (100, 100, 100))
                        visualization_tool.visualise_layer_without_hooks(diagnosis_path, input_nii.affine)

            output_pt = best_decoder(input_pt)
            output_np = output_pt.detach().numpy()[0][0]
            output_nii = nib.Nifti1Image(output_np, affine=input_nii.affine)

            nib.save(input_nii, path.join(diagnosis_path, 'input.nii'))
            nib.save(output_nii, path.join(diagnosis_path, 'output.nii'))


if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        print("unknown arguments: %s" % parser.parse_known_args()[1])
    main(options)
