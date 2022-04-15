import torch
from torch.nn import MSELoss


def VAELoss(input, reconstruction, mu, logVar):
    kl_divergence = (
        0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp()) / mu.shape[0]
    )
    # recon_error = MSELoss(reduction="mean")(reconstruction, input)
    recon_error = MSELoss(reduction="sum")(reconstruction, input)
    # recon_error = torch.sum((reconstruction - input) ** 2) / input.shape[0]
    return recon_error, kl_divergence


def multiply_list(L, factor):
    product = 1
    for x in L:
        product = product * x / factor
    return product
