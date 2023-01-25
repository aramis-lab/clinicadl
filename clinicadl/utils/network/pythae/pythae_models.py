from clinicadl.utils.network.pythae.pythae_utils import BasePythae


class pythae_VAE(BasePythae):
    def __init__(
        self,
        input_size,
        latent_space_size,
        feature_size,
        n_conv,
        io_layer_channels,
        gpu=False,
    ):
        super(pythae_VAE, self).__init__(
            input_size=input_size,
            latent_space_size=latent_space_size,
            feature_size=feature_size,
            n_conv=n_conv,
            io_layer_channels=io_layer_channels,
            gpu=gpu
        )
        self.input_size = input_size
        self.latent_space_size = latent_space_size

    def compute_outputs_and_loss(self, input_dict, criterion, use_labels=False):
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        x = input_dict["image"].to(self.device)
        recon_x, mu, log_var, z = self.forward(x)
        losses = self.get_model().loss_function(recon_x, x, mu, log_var, z)
        loss_dict = {
            "loss": losses[0],
            "recon_loss": losses[1],
            "kl_loss": losses[2],
        }
        return recon_x, loss_dict

    def get_model(self):
        from pythae.models import VAE, VAEConfig
        model_config = VAEConfig(
            input_dim=self.input_size,
            latent_dim=self.latent_space_size
        )
        return VAE(
            model_config=model_config,
            encoder=self.encoder,
            decoder=self.decoder,
        )


class pythae_BetaVAE(BasePythae):
    def __init__(
        self,
        input_size,
        latent_space_size,
        feature_size,
        n_conv,
        io_layer_channels,
        beta,
        gpu=False,
    ):
        super(pythae_BetaVAE, self).__init__(
            input_size=input_size,
            latent_space_size=latent_space_size,
            feature_size=feature_size,
            n_conv=n_conv,
            io_layer_channels=io_layer_channels,
            gpu=gpu
        )
        self.input_size = input_size
        self.latent_space_size = latent_space_size
        self.beta = beta

    def compute_outputs_and_loss(self, input_dict, criterion, use_labels=False):
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        x = input_dict["image"].to(self.device)
        recon_x, mu, log_var, z = self.forward(x)
        losses = self.get_model().loss_function(recon_x, x, mu, log_var, z)
        loss_dict = {
            "loss": losses[0],
            "recon_loss": losses[1],
            "kl_loss": losses[2],
        }
        return recon_x, loss_dict

    def get_model(self):
        from pythae.models import BetaVAE, BetaVAEConfig
        model_config = BetaVAEConfig(
            input_dim=self.input_size,
            latent_dim=self.latent_space_size
            beta=self.beta
        )
        return BetaVAE(
            model_config=model_config,
            encoder=self.encoder,
            decoder=self.decoder,
        )


class pythae_VAE_LinNF(BasePythae):
    def __init__(
        self,
        input_size,
        latent_space_size,
        feature_size,
        n_conv,
        io_layer_channels,
        flows,
        gpu=False,
    ):
        super(pythae_VAE_LinNF, self).__init__(
            input_size=input_size,
            latent_space_size=latent_space_size,
            feature_size=feature_size,
            n_conv=n_conv,
            io_layer_channels=io_layer_channels,
            gpu=gpu
        )
        self.input_size = input_size
        self.latent_space_size = latent_space_size
        self.flows = flows

    def compute_outputs_and_loss(self, input_dict, criterion, use_labels=False):
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        x = input_dict["image"].to(self.device)
        recon_x, mu, log_var, z = self.forward(x)
        losses = self.get_model().loss_function(recon_x, x, mu, log_var, z)
        loss_dict = {
            "loss": losses[0],
            "recon_loss": losses[1],
            "kl_loss": losses[2],
        }
        return recon_x, loss_dict

    def get_model(self):
        from pythae.models import BetaVAE, BetaVAEConfig
        model_config = BetaVAEConfig(
            input_dim=self.input_size,
            latent_dim=self.latent_space_size
            flows=self.flows
        )
        return BetaVAE(
            model_config=model_config,
            encoder=self.encoder,
            decoder=self.decoder,
        )

