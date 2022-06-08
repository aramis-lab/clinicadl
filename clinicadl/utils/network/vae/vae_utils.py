import torch
from torch.nn import BCELoss, MSELoss


def VAEGaussianLoss(input, reconstruction, mu, logVar):
    kl_divergence = (
        0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp()) / mu.shape[0]
    )
    recon_error = MSELoss(reduction="sum")(reconstruction, input)
    return recon_error, kl_divergence


def VAEBernoulliLoss(input, reconstruction, mu, logVar):
    kl_divergence = (
        0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp()) / mu.shape[0]
    )
    recon_error = BCELoss(reduction="sum")(reconstruction, input)
    return recon_error, kl_divergence


def VAEContinuousBernoulliLoss(input, reconstruction, mu, logVar):
    kl_divergence = (
        0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp()) / mu.shape[0]
    )
    recon_error = BCELoss(reduction="sum")(reconstruction, input)
    log_constant = sumlogC(reconstruction)
    return kl_divergence, recon_error, log_constant


def sumlogC(x, eps=1e-5):
    """
    Numerically stable implementation of
    sum of logarithm of Continous Bernoulli
    constant C, using Taylor 2nd degree approximation

    Parameter
    ----------
    x : Tensor of dimensions (batch_size, dim)
        x takes values in (0,1)
    """
    x = torch.clamp(x, eps, 1.0 - eps)
    mask = torch.abs(x - 0.5).ge(eps)
    far = torch.masked_select(x, mask)
    close = torch.masked_select(x, ~mask)
    far_values = torch.log((torch.log(1.0 - far) - torch.log(far)).div(1.0 - 2.0 * far))
    close_values = torch.log(torch.tensor((2.0))) + torch.log(
        1.0 + torch.pow(1.0 - 2.0 * close, 2) / 3.0
    )
    return far_values.sum() + close_values.sum()


def multiply_list(L, factor):
    product = 1
    for x in L:
        product = product * x / factor
    return product
