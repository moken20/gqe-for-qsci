import os
import functools

import torch
import hydra, wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from gqe_qsci.train_pipeline import TrainPipeline
from gqe_qsci.factory import Factory

torch.load = functools.partial(torch.load, weights_only=False)

os.environ["OMP_NUM_THREADS"] = "1"
torch.set_float32_matmul_precision("medium")


def setup_callbacks(cfg):
    callbacks = [
        ModelCheckpoint(
            dirpath=f"{cfg.output}/models",
            filename="{epoch:02d}",
            save_top_k=1,
            monitor="epoch",
            mode="max",
            save_last=True,
            every_n_epochs=cfg.trainer.checkpoint_every_n_iters,
            enable_version_counter=False
        ),
    ]
    return callbacks


def get_checkpoint_path(cfg):
    if not cfg.trainer.load_checkpoint:
        print("No checkpoint loading requested. Training from scratch.")
        return None
    path = f"{cfg.output}/models/last.ckpt"
    if os.path.exists(path):
        print(f"Loading checkpoint from {path}")
        return path
    print(f"Checkpoint {path} does not exist. Training from scratch.")
    return None

def setup_logger(cfg):
    save_dir = cfg.output
    run_id_file = os.path.join(save_dir, "run_id")
    run_id = None
    if os.path.exists(run_id_file):
        with open(run_id_file, "r") as f:
            run_id = f.read().strip()
        print(f"Resuming training from run_id: {run_id}")
    logger = WandbLogger(
        project=cfg.wandb.project,
        settings=wandb.Settings(init_timeout=300),
        config=OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            ),
        save_dir=save_dir,
        name=cfg.wandb.name,
        id=run_id if run_id is not None else None,
        resume='allow'
    )
    if not os.path.exists(run_id_file): 
        run_id = logger.experiment.id
        with open(run_id_file, "w") as f:
            f.write(str(run_id))

    return logger


def train(cfg):
    callbacks = setup_callbacks(cfg)
    logger = setup_logger(cfg)
    training = TrainPipeline(Factory(), cfg)
    training.set_seed(cfg.trainer.seed)
    trainer = Trainer(
        logger=logger if logger is not None else False,
        callbacks=callbacks,
        max_epochs=cfg.trainer.max_iters,
        deterministic=True, 
        reload_dataloaders_every_n_epochs=1,
        precision="16-mixed",
        devices=1,
        log_every_n_steps=10
    )
    trainer.fit(training, ckpt_path=get_checkpoint_path(cfg))

@hydra.main(version_base="1.3", config_path="configs", config_name="default")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()