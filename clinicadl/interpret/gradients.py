import torch
from torch.cuda.amp import autocast

from clinicadl.utils.exceptions import ClinicaDLArgumentError


class VanillaBackProp:
    """
    Produces gradients generated with vanilla back propagation from the image
    """

    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device

    def generate_gradients(self, input_batch, target_class, amp=False, **kwargs):
        # Forward
        input_batch = input_batch.to(self.device)
        input_batch.requires_grad = True
        with autocast(enabled=amp):
            if hasattr(self.model, "variational") and self.model.variational:
                _, _, _, model_output = self.model(input_batch)
            else:
                model_output = self.model(input_batch)
        # Target for backprop
        one_hot_output = torch.zeros_like(model_output)
        one_hot_output[:, target_class] = 1
        one_hot_output = one_hot_output.to(self.model.device)
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        gradients = input_batch.grad.cpu()
        return gradients


class GradCam:
    """
    Produces Grad-CAM to a monai.networks.nets.Classifier
    """

    def __init__(self, model):
        from clinicadl.utils.network.sub_network import CNN

        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device

        if not isinstance(model, CNN):
            raise ValueError("Grad-CAM was only implemented for CNN models.")

    def generate_gradients(
        self, input_batch, target_class, level=None, amp=False, **kwargs
    ):
        """
        Generate the gradients map corresponding to the input_tensor.
        Args:
            input_batch (Tensor): tensor representing a batch of images.
            target_class (int): allows to choose from which node the gradients are back-propagated.
                Default will back-propagate from the node corresponding to the true class of the image.
            level (int): layer number in the convolutional part after which the feature map is chosen.
            amp (bool): whether or not to use automatic mixed precision during forward.
        Returns:
            (Tensor): the gradients map
        """
        if len(input_batch.shape) == 4:
            mode = "bilinear"
        elif len(input_batch.shape) == 5:
            mode = "trilinear"
        else:
            raise ClinicaDLArgumentError(
                "Input batch should be 4D or 5D to correspond to 2D or 3D images."
            )

        input_batch = input_batch.to(self.device)

        # Dissect model
        if level is None or level >= len(self.model.convolutions):
            conv_part = self.model.convolutions
            pre_fc_part = torch.nn.Identity()
            fc_part = self.model.fc
        else:
            conv_part = self.model.convolutions[:level]
            pre_fc_part = self.model.convolutions[level:]
            fc_part = self.model.fc

        # Get last conv feature map
        feature_maps = conv_part(input_batch).detach()
        feature_maps.requires_grad = True
        with autocast(enabled=amp):
            model_output = fc_part(pre_fc_part(feature_maps))
        # Target for backprop
        one_hot_output = torch.zeros_like(model_output)
        if target_class is not None:
            one_hot_output[:, target_class] = 1
        else:
            labels = input_batch["label"]
            for i, target_class in enumerate(labels):
                one_hot_output[i, target_class] = 1
        one_hot_output = one_hot_output.to(self.device)
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        gradients = feature_maps.grad
        pooled_gradients = torch.mean(gradients, dim=[2, 3]).unsqueeze(2).unsqueeze(3)

        # Weight feature maps according to pooled gradients
        feature_maps.requires_grad = False
        feature_maps *= pooled_gradients
        # Take the mean of all weighted feature maps
        grad_cam = torch.mean(feature_maps, dim=1).cpu().unsqueeze(1)
        resize_transform = torch.nn.Upsample(
            input_batch.shape[2::], mode=mode, align_corners=True
        )

        return resize_transform(grad_cam)


method_dict = {"gradients": VanillaBackProp, "grad-cam": GradCam}
