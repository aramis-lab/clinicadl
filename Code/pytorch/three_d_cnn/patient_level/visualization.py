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


def main(options):

    model = eval(options.model)()
    decoder = Decoder(model)
    best_decoder, _ = load_model(decoder, options.model_path)
    sets = ['train', 'validation']

    for set in sets:

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
