# `clinicadl hugging-face` - Share your experiment with hugging-face

[Hugging Face](https://huggingface.co) is a machine learning (ML) and data science platform and community that helps users build, deploy and train machine learning models.

It provides the infrastructure to demo, run and deploy artificial intelligence (AI) in live applications. Users can also browse through models and data sets that other people have uploaded. 
Hugging Face is often called the GitHub of machine learning because it lets developers share and test their work openly.

For now, you can push and pull [MAPS](../Introduction.md#maps-definition) in Hugging Face with clinicadl.

You can find the ClinicaDL organization [here](https://huggingface.co/ClinicaDL) and send a request to join and access our pre-trained models.

!!! warning "Sharing limitations"
    Depending on the data, you can not be authorized to share all the MAPS, 
    because it includes TSV files with patient informations.

## `push` - Push your MAPS to Hugging Face 

### Description

This commnandline will create a new repository in Hugging face or update the repository with the given name. 

### Running the task

```bash
clinicadl hugging-face push [OPTIONS] ORGANIZATION MAPS_DIRECTORY HF_MAPS_DIRECTORY
```
where:

  - `ORGANIZATION` (Path) is either the name of the organization or your profile.
  - `MAPS_DIRECTORY` (Path) is the folder you want to push, containing the dataset in a MAPS hierarchy.
  - `HF_MAPS_DIRECtORY` (Path) is the name you want the repo to have in hugging face.


#### Example of how to run the task :

```bash
clinicadl hugging-face push clinicadl /DATA/maps maps-test
```

## `pull` - Pull a MAPS from Hugging Face 

### Description

This commnandline will download a new repository from Hugging face. 

### Running the task

```bash
clinicadl hugging-face pull [OPTIONS] ORGANIZATION MAPS_DIRECTORY HF_MAPS_DIRECTORY
```
where:

  - `ORGANIZATION` (Path) is either the name of the organization or your profile.
  - `MAPS_DIRECTORY` (Path) is the folder you want to push, containing the dataset in a MAPS hierarchy.
  - `HF_MAPS_DIRECtORY` (Path) is the name you want the repo to have in hugging face.


#### Example of how to run the task :

```bash
clinicadl hugging-face pull clinicadl /DATA/maps maps-test
```

This commandline will download in a new repository (`DATA/maps`) the MAPS named `maps-test` in the clinicadl organization in Hugging Face.

### Output tree
