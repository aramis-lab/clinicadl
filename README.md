# Clinica Deep Learning AD
This repository contains a software framework for reproducible experiments with 2D convolutional neural network on automatic classification of Alzheimer's disease (AD) using anatomical MRI data from the publicly available datasets ADNI. It is developed by Junhao WEN and Ouyang WEI.
This architecture relies heavily on the Clinica software platform that you will need to install. Another prerequisite is to do image processing for the original MRI data by using Clinica, then we fit the processed data into this CNN.

# Projects
- 1) Implement 2D CNN to fit the processed png images from MRI
- 2) Compress the 3D MRI to 2D multi-chanel image (compressive sensing), then fit into the 2D CNN.

# Dependencies:
- 1) Clinica
- 2) TensorFlow

# Example:
To run this example, you should run this command in your terminal:
# run training: the hyperparameters are listed in **adni_png_train.py**
```
python run_train.py --max_steps 10000 --dropout_rate 0.2
```
# run testing:
```
python run_test.py
```