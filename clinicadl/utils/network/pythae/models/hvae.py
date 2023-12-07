from clinicadl.utils.network.pythae.pythae_utils import BasePythae


class pythae_HVAE(BasePythae):
    def __init__(
        self,
        encoder_decoder_config,
        n_lf,
        eps_lf,
        beta_zero,
        gpu=False,
    ):

        from pythae.models import HVAE, HVAEConfig

        encoder, decoder = super(pythae_HVAE, self).__init__(
            encoder_decoder_config = encoder_decoder_config,
            gpu=gpu,
        )

        model_config = HVAEConfig(
            input_dim=self.input_size,
            latent_dim=self.latent_space_size,
            n_lf=n_lf,
            eps_lf=eps_lf,
            beta_zero=beta_zero,
        )
        self.model = HVAE(
            model_config=model_config,
            encoder=encoder,
            decoder=decoder,
        )
    
    def get_trainer_config(self, output_dir, num_epochs, learning_rate, batch_size, optimizer):
        from pythae.trainers import BaseTrainerConfig
        return BaseTrainerConfig(
            output_dir=output_dir,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            optimizer_cls=optimizer,
        )
