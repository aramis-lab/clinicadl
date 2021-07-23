!!! warning "Coming soon"
    Tensor generation was not added yet to the command line.


## Outputs

Results are stored in the MAPS of path `model_path`, according to
the following file system:
```
<model_path>
    ├── fold-0  
    ├── ...  
    └── fold-<fold>
        └── best-<metric>
                └── <data_group>
                    └── tensors
                        ├── <participant_id>_<session_id>_{image|patch|roi|slice}-<i>_input.pt
                        └── <participant_id>_<session_id>_{image|patch|roi|slice}-<i>_output.pt
```
For each `participant_id`, `session_id` and index of the part of the image (`i`),
the input and the output tensors are saved in.