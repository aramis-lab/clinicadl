from clinicadl.utils.network.pythae.pythae_utils import BasePythae


class pythae_RAE_L2(BasePythae):
    def __init__(
        self,
        input_size,
        latent_space_size,
        feature_size,
        n_conv,
        io_layer_channels,
        embedding_weight,
        reg_weight,
        gpu=False,
    ):

        from pythae.models import RAE_L2, RAE_L2_Config

        encoder, decoder = super(pythae_RAE_L2, self).__init__(
            input_size=input_size,
            latent_space_size=latent_space_size,
            feature_size=feature_size,
            n_conv=n_conv,
            io_layer_channels=io_layer_channels,
            gpu=gpu,
            is_ae=True,
        )

        model_config = RAE_L2_Config(
            input_dim=self.input_size,
            latent_dim=self.latent_space_size,
            embedding_weight=embedding_weight,
            reg_weight=reg_weight,
        )
        self.model = RAE_L2(
            model_config=model_config,
            encoder=encoder,
            decoder=decoder,
        )

    def get_trainer_config(self, output_dir, num_epochs, learning_rate, batch_size):
        from pythae.trainers import CoupledOptimizerTrainerConfig
        return CoupledOptimizerTrainerConfig(
            output_dir=output_dir,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size
        )