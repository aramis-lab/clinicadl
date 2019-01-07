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
parser.add_argument("model", type=str, choices=["Conv_3", "Conv_4", "Test", "Test_nobatch", "Rieke", "Test2", 'Optim'],
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


def main(options):

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
