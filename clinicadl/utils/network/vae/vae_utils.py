import torch
from torch.nn import MSELoss


def VAELoss(input, reconstruction, mu, logVar):
    kl_divergence = (
        0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp()) / mu.shape[1]
    )
    recon_error = MSELoss(reduction="mean")(reconstruction, input)
    # recon_error = torch.sum((reconstructed - input_) ** 2) / input_.shape[0]
    return recon_error, kl_divergence
