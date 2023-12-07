RS_PYTHAE_DICT = {
    "pythae_BetaVAE": {
        "beta": "choice",
    },
    "pythae_VAE_LinNF": {
        "flows": "choice",
    },
    "pythae_VAE_IAF": {
        "n_made_blocks": "randint",
        "n_hidden_in_made": "randint",
        "hidden_size": "choice",
    },
    "pythae_BetaTCVAE": {
        "beta": "choice",
        "alpha": "choice",
        "gamma": "choice",
    },
    "pythae_MSSSIM_VAE": {
        # MS SSIM VAE
        "beta": "choice",
        "window_size": "choice",
    },
    "pythae_VQVAE": {
        # VQ VAE
        "commitment_loss_factor": "choice",
        "quantization_loss_factor": "choice",
        "num_embeddings": "choice",
        "use_ema": "choice",
        "decay": "choice",
    },
    "pythae_RAE_L2": {
        # Regularized AE with L2 decoder param
        "embedding_weight": "choice",
        "reg_weight": "choice",
    },
    "pythae_RAE_GP": {
        # Regularized AE with gradient penalty
        "embedding_weight": "choice",
        "reg_weight": "choice",
    },
    "pythae_Adversarial_AE": {
        # Adversarial AE
        "adversarial_loss_scale": "choice",
    },
    "pythae_VAEGAN": {
        # VAE GAN
        "adversarial_loss_scale": "choice",
        "reconstruction_layer": "choice",
        "margin": "choice",
        "equilibrium": "choice",
    },
    "pythae_WAE_MMD": {
        # Wasserstein Autoencoder
        "kernel_choice": "choice",
        "reg_weight": "choice",
        "kernel_bandwidth": "choice",
    },
    "pythae_INFOVAE_MMD": {
        # Info VAE
        "kernel_choice": "choice",
        "alpha": "choice",
        "lbd": "choice",
        "kernel_bandwidth": "choice",
    },
}
