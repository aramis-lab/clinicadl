from clinicadl.utils.network.pythae.pythae_utils import BasePythae


class pythae_INFOVAE_MMD(BasePythae):
    def __init__(
        self,
        input_size,
        latent_space_size,
        feature_size,
        n_conv,
        io_layer_channels,
        kernel_choice,
        alpha,
        lbd,
        kernel_bandwidth,
        gpu=False,
    ):

        from pythae.models import INFOVAE_MMD, INFOVAE_MMD_Config

        encoder, decoder = super(pythae_INFOVAE_MMD, self).__init__(
            input_size=input_size,
            latent_space_size=latent_space_size,
            feature_size=feature_size,
            n_conv=n_conv,
            io_layer_channels=io_layer_channels,
            gpu=gpu
        )

        model_config = INFOVAE_MMD_Config(
            input_dim=self.input_size,
            latent_dim=self.latent_space_size,
            kernel_choice=kernel_choice,
            alpha=alpha,
            lbd=lbd,
            kernel_bandwidth=kernel_bandwidth,
        )
        self.model = INFOVAE_MMD(
            model_config=model_config,
            encoder=encoder,
            decoder=decoder,
        )

    def get_trainer_config(self, output_dir, num_epochs, learning_rate, batch_size):
        from pythae.trainers import BaseTrainerConfig
        return BaseTrainerConfig(
            output_dir=output_dir,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size
        )
