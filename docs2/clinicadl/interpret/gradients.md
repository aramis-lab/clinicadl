# Gradients

[Clinicadl Index](../../README.md#clinicadl-index) /
[Clinicadl](../index.md#clinicadl) /
[Interpret](./index.md#interpret) /
Gradients

> Auto-generated documentation for [clinicadl.interpret.gradients](../../../clinicadl/interpret/gradients.py) module.

- [Gradients](#gradients)
  - [GradCam](#gradcam)
    - [GradCam().generate_gradients](#gradcam()generate_gradients)
  - [VanillaBackProp](#vanillabackprop)
    - [VanillaBackProp().generate_gradients](#vanillabackprop()generate_gradients)

## GradCam

[Show source in gradients.py:35](../../../clinicadl/interpret/gradients.py#L35)

Produces Grad-CAM to a monai.networks.nets.Classifier

#### Signature

```python
class GradCam:
    def __init__(self, model):
        ...
```

### GradCam().generate_gradients

[Show source in gradients.py:50](../../../clinicadl/interpret/gradients.py#L50)

Generate the gradients map corresponding to the input_tensor.

#### Arguments

- `input_batch` *Tensor* - tensor representing a batch of images.
- `target_class` *int* - allows to choose from which node the gradients are back-propagated.
    Default will back-propagate from the node corresponding to the true class of the image.
- `level` *int* - layer number in the convolutional part after which the feature map is chosen.

#### Returns

- `(Tensor)` - the gradients map

#### Signature

```python
def generate_gradients(self, input_batch, target_class, level=None, **kwargs):
    ...
```



## VanillaBackProp

[Show source in gradients.py:6](../../../clinicadl/interpret/gradients.py#L6)

Produces gradients generated with vanilla back propagation from the image

#### Signature

```python
class VanillaBackProp:
    def __init__(self, model):
        ...
```

### VanillaBackProp().generate_gradients

[Show source in gradients.py:16](../../../clinicadl/interpret/gradients.py#L16)

#### Signature

```python
def generate_gradients(self, input_batch, target_class, **kwargs):
    ...
```