import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from fashion_mnist_classifier.data.dataset import FashionMNISTDataModule
from fashion_mnist_classifier.utils.helpers import ensure_data_available, get_git_commit_id


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    print("Start Fashion-MNIST Training")
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed, workers=True)

    ensure_data_available(cfg.data.data_dir)

    print("\nSetting up DataModule...")
    dm = FashionMNISTDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        val_split=cfg.data.val_split,
        num_workers=cfg.data.num_workers,
    )

    print(f"\nInitializing model: {cfg.model._target_}")
    model = hydra.utils.instantiate(cfg.model)

    print(f"\nSetting up MLflow tracking: {cfg.logging.tracking_uri}")
    mlf_logger = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        tracking_uri=cfg.logging.tracking_uri,
        run_name=cfg.logging.run_name,
    )

    mlf_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    mlf_logger.log_hyperparams({"git_commit_id": get_git_commit_id()})

    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="fashion-mnist-{epoch:02d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(monitor="val/loss", patience=5, mode="min", verbose=True),
    ]

    print("\nInitializing PyTorch Lightning Trainer...")
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        logger=mlf_logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.training.log_every_n_steps,
        gradient_clip_val=cfg.training.gradient_clip_val,
        deterministic=cfg.training.deterministic,
    )

    print("\nStart training...")
    trainer.fit(model, dm)

    print("\nRun test evaluation...")
    trainer.test(model, dm, ckpt_path="best")

    print("\nTraining complete!")
    print(f"MLflow: {cfg.logging.tracking_uri}")
    print("Checkpoints: checkpoints/")


if __name__ == "__main__":
    train()
