import torch


class VanillaBackProp:
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model, gpu=False):
        self.model = model
        self.gradients = None
        self.model.eval()
        self.gpu = gpu

        if gpu:
            self.model = self.model.cuda()

    def generate_gradients(self, input_batch, target_class):
        # Forward
        input_batch.requires_grad = True
        if hasattr(self.model, "variational") and self.model.variational:
            _, _, _, model_output = self.model(input_batch)
        else:
            model_output = self.model(input_batch)
        # Target for backprop
        one_hot_output = torch.zeros_like(model_output)
        one_hot_output[:, target_class] = 1
        if self.gpu:
            one_hot_output = one_hot_output.cuda()
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        gradients_as_arr = input_batch.grad.cpu().numpy()
        return gradients_as_arr
