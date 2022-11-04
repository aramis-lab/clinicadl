#! /usr/bin/env python3
# -*- coding: utf-8 -*-

#
# @author Vladimir S. FONOV
# @date 13/04/2018
import argparse
import os
from re import sub
import sys

import numpy as np
import io
import copy

from aqc_data import *
from model.util import *

import torch
import torch.nn as nn

from torch.autograd import Variable


default_data_dir=os.path.dirname(sys.argv[0])
if default_data_dir=='' or default_data_dir is None: default_data_dir='.'

def parse_options():

    parser = argparse.ArgumentParser(description='Apply automated QC',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--ref", action="store_true", default=False,
                        help="Use reference images")
    parser.add_argument("--image", type=str,
                        help="Input image prefix: <prefix>_{0,1,2}.jpg")
    parser.add_argument("--volume", type=str,
                        help="Input minc volume (need minc2 simple)")
    parser.add_argument("--resample", type=str,
                        help="Resample to standard space using provided xfm, needs mincresample")
    parser.add_argument("--low", type=float,
                        help="Winsorize intensities in the input image", default=5.0)
    parser.add_argument("--high", type=float, 
                        help="Winsorize intensities in the input image", default=95.0)
    parser.add_argument("--load", type=str, default=None,
                        help="Load pretrained model (will try to  load from {}".format(default_data_dir))
    parser.add_argument("--net", choices=['r18', 'r34', 'r50','r101','r152','sq101'],
                        help="Network type",default='r18')
    parser.add_argument('--raw', action="store_true", default=False,
                        help='Print raw score [0:1]')
    parser.add_argument('-q', '--quiet', action="store_true",default=False,   
                        help='Quiet mode, set status code to 0 - Pass, 1 - fail')
    parser.add_argument('--batch', type=str,   
                        help='Process minc files in batch mode, provide list of files')
    parser.add_argument('--batch-size', type=int, default=1, 
                        dest='batch_size',
                        help='Batch size in batch mode')
    parser.add_argument('--batch-workers', type=int, default=1,
                        dest='batch_workers',
                        help='Number of workers in batch mode')
    parser.add_argument('--batch_pics', action="store_true", default=False,
                        help='Process QC pics in batch mode instead of MINC volumes')
    parser.add_argument('--gpu', action="store_true", default=False,
                        help='Run inference in gpu')
    parser.add_argument("--dist",action="store_true",default=False,
                        help="Predict misregistration distance instead of class membership")
    parser.add_argument("--freesurfer",default=None,
                        help="Process freesurfer output from recon all, provide subject directory, need mri_convert and nibabel ")
    parser.add_argument('--debug', action="store_true", default=False,
                        help='Print debug messages')

    params = parser.parse_args()
    
    return params


if __name__ == '__main__':
    params = parse_options()
    use_ref = params.ref


    if params.load is None:

        if params.dist:
            params.load = default_data_dir + os.sep \
                + 'dist' + os.sep \
                + 'model_dist_' + params.net + ('_ref' if params.ref else '') + os.sep + \
                'best_loss.pth' 
        else:
            params.load = default_data_dir + os.sep \
                + 'cls' + os.sep \
                + 'model_' + params.net + ('_ref' if params.ref else '') + os.sep + \
                'best_tnr.pth'

    if not os.path.exists(params.load):
       print("Missing model:",params.load,file=sys.stderr)
       exit(100)

    model = get_qc_model(params, use_ref=use_ref, predict_dist=params.dist)
    model.train(False)

    if params.gpu:
        model=model.cuda()


    with torch.no_grad():
        if params.batch is not None:
            if not params.batch_pics :
                dataset = MincVolumesDataset(csv_file=params.batch,
                    winsorize_low=params.low,
                    winsorize_high=params.high,
                    data_prefix=default_data_dir + "/../data",
                    use_ref=use_ref)
            else:
                dataset = QCImagesDataset(csv_file=params.batch,
                            data_prefix=default_data_dir + "/../data",
                            use_ref=use_ref)

            dataloader = DataLoader(dataset,
                            batch_size=params.batch_size,
                            shuffle=False,
                            num_workers=params.batch_workers)

            for i_batch, sample_batched in enumerate(dataloader):
                inputs, files = sample_batched
                if params.gpu: inputs = inputs.cuda()
                outputs = model(inputs)
                if params.gpu: outputs = outputs.cpu()
                
                if params.dist:
                    if params.raw:
                        preds   = outputs[:,0].tolist()
                    else:
                        preds   = torch.max(outputs, 1)[1].tolist()

                    for i,j in zip(files,preds):
                        print("{},{}".format(i,j))
                else:
                    outputs = nn.functional.softmax(outputs,1)
                    if params.raw:
                        preds   = outputs[:,1].tolist()
                    else:
                        preds   = torch.max(outputs, 1)[1].tolist()

                    for i,j in zip(files,preds):
                        print("{},{}".format(i,j))
        else:
            if params.image is not None:
                inputs = load_qc_images( [params.image+'_0.jpg', params.image+'_1.jpg', params.image+'_2.jpg'])
            elif params.volume is not None:
                tmpdir=None
                volume=params.volume

                if params.resample is not None:
                    import tempfile,subprocess,shutil
                    tmpdir=tempfile.mkdtemp(prefix='deep_qc')
                    tmp_vol=tmpdir+os.sep+'tmp.mnc'
                    # provide sampling in the standard space
                    try:
                        args=['mincresample', '-q' ,'-transform',params.resample,
                            '-dircos', '1' ,'0', '0','0', '1', '0', '0', '0', '1', 
                            '-step', '1', '1', '1',
                            '-start', '-96', '-132', '-78',
                            '-nelements', '193', '229', '193', params.volume, tmp_vol,'-tfm_input_sampling']
                        subprocess.check_call(args,stdout=subprocess.DEVNULL if not params.debug else None,stderr=subprocess.DEVNULL if not params.debug else None)
                    except:
                        shutil.rmtree(tmpdir)
                        raise
                    volume=tmp_vol
                
                inputs = load_minc_images(volume,winsorize_low=params.low,winsorize_high=params.high)
                if tmpdir is not None:
                    shutil.rmtree(tmpdir)
            elif params.freesurfer is not None:
                in_mgz=params.freesurfer+os.sep+'mri'+os.sep+'orig.mgz'
                in_xfm=params.freesurfer+os.sep+'mri'+os.sep+'transforms'+os.sep+'talairach.xfm'
                if not os.path.exists(in_mgz):
                    print("Missing freesurfer file:{}".format(in_mgz),file=sys.stderr)
                    exit(1)
                if not os.path.exists(in_xfm):
                    print("Missing freesurfer file:{}".format(in_xfm),file=sys.stderr)
                    exit(1)
                import tempfile,subprocess,shutil
                try:
                    tmpdir=tempfile.mkdtemp(prefix='deep_qc')
                    tmp_vol=tmpdir+os.sep+'tmp.mgh'
                    args=['mri_convert', in_mgz , 
                          '--apply_transform',in_xfm, 
                          '-oc', '0', '0', '0', 
                          '-vs','1', '1', '1',
                          tmp_vol]
                    subprocess.check_call(args,stdout=subprocess.DEVNULL if not params.debug else None,stderr=subprocess.DEVNULL if not params.debug else None)
                    inputs = load_talairach_mgh_images(tmp_vol,winsorize_low=params.low,winsorize_high=params.high)
                    # from skimage import io
                    # for i,j in enumerate(inputs):
                    #     io.imsave(f"debug_{i}.jpg",np.clip( ((inputs[i]+0.5)*255).squeeze().numpy(),0,255).astype('uint8'))
                finally:
                    shutil.rmtree(tmpdir)
            else:
                print("Specify input volume or image prefix or batch list, see help with --help",file=sys.stderr)
                exit(1)
            
            if use_ref:
               ref_inputs = load_qc_images(
                           [ default_data_dir + "/../data/" + "mni_icbm152_t1_tal_nlin_sym_09c_0.jpg",
                             default_data_dir + "/../data/" + "mni_icbm152_t1_tal_nlin_sym_09c_1.jpg",
                             default_data_dir + "/../data/" + "mni_icbm152_t1_tal_nlin_sym_09c_2.jpg" ])

               inputs = torch.cat( [ item for sublist in zip(inputs, ref_inputs) for item in sublist ] ).unsqueeze_(0)
            else:
              # convert inputs into properly formated tensor
              # with a single batch dimension
              inputs = torch.cat( inputs ).unsqueeze_(0)

            outputs = model(inputs)

            if params.dist:
                outputs = outputs[0,0]
                preds   = (outputs>10.0).squeeze() # TODO : parametrize threshold
            else:
                outputs = nn.functional.softmax(outputs,1)
                preds   = torch.max(outputs,1)[1].squeeze()
                outputs = outputs[0,1]
            # raw score
            if params.raw:
                print(float(outputs))
            elif not params.quiet:
                if preds:
                    print("Pass")
                else:
                    print("Fail")
            else:
                exit(0 if preds[0]==1 else 1)

