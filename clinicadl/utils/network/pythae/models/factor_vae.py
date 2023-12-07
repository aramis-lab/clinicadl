from clinicadl.utils.network.pythae.pythae_utils import BasePythae


class pythae_FactorVAE(BasePythae):
    def __init__(
        self,
        encoder_decoder_config,
        gamma,
        gpu=False,
    ):

        from pythae.models import FactorVAE, FactorVAEConfig

        encoder, decoder = super(pythae_FactorVAE, self).__init__(
            encoder_decoder_config = encoder_decoder_config,
            gpu=gpu,
        )

        model_config = FactorVAEConfig(
            input_dim=self.input_size,
            latent_dim=self.latent_space_size,
            gamma=gamma,
        )
        self.model = FactorVAE(
            model_config=model_config,
            encoder=encoder,
            decoder=decoder,
        )

    def get_trainer_config(self, output_dir, num_epochs, learning_rate, batch_size, optimizer):
        from pythae.trainers import AdversarialTrainerConfig
        return AdversarialTrainerConfig(
            output_dir=output_dir,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            optimizer_cls=optimizer,
        )
