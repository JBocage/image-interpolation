from trainers.auto_encoder_trainer import AutoencoderTrainer

trainer = AutoencoderTrainer(
    trainer_name="test",
    overwrite_checkpoint=True,

)

trainer.train(
    batch_size=32,
    epochs=10
)