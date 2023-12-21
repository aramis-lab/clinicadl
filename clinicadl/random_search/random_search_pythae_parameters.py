RS_PYTHAE_DICT = {
    "pythae_BetaVAE": {
        "beta": "choice",
    },
    "pythae_VAE_LinNF": {
        "flows": "choice",
    },
    "pythae_VAE_IAF": {
        "n_made_blocks": "choice",
        "n_hidden_in_made": "choice",
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
    "pythae_CIWAE": {
        "beta": "choice",
        "number_samples": "randint",
    },
    "pythae_PIWAE": {
        "number_gradient_estimates": "randint",
        "number_samples": "randint",
    },
    "pythae_MIWAE": {
        "number_gradient_estimates": "randint",
        "number_samples": "randint",
    },
    "pythae_IWAE": {
        "number_samples": "choice",
    },
    "pythae_DisentangledBetaVAE": {
        "beta": "choice",
        "C": "choice",
        "warmup_epoch": "choice",
    },
    "pythae_FactorVAE": {
        "gamma": "choice",
    },
    "pythae_SVAE": {
    },
    "pythae_PoincareVAE": {
        "prior_distribution": "choice",
        "posterior_distribution": "choice",
        "curvature": "choice",
    },
    "pythae_HVAE": {
        "n_lf": "choice",
        "eps_lf": "choice",
        "beta_zero": "choice",
    },
    "pythae_RHVAE": {
        "n_lf": "choice",
        "eps_lf": "choice",
        "beta_zero": "choice",
        "temperature": "choice",
        "regularization": "choice",
    },
    "pythae_VAMP": {
        "number_components": "choice",
        "linear_scheduling_steps": "choice",
    },
    "pythae_MSSSIM_VAE": {
        "beta": "choice",
        "window_size": "choice",
    }
}
