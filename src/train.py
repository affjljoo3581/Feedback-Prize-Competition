import argparse
import os
import warnings

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from lightning import FeedbackPrizeDataModule, FeedbackPrizeModule

try:
    import apex

    amp_backend = apex.__name__
except ModuleNotFoundError:
    amp_backend = "native"

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(config: DictConfig):
    module = FeedbackPrizeModule(config)
    datamodule = FeedbackPrizeDataModule(config)

    name = f"{config.train.name}-fold{config.dataset.fold_index}"
    checkpoint = ModelCheckpoint(
        monitor="val/f1_score",
        mode="max",
        save_weights_only=True,
        save_last=not config.train.save_best_checkpoint,
    )

    Trainer(
        gpus=config.train.gpus,
        logger=WandbLogger(project="feedback-prize-2021", name=name),
        callbacks=[checkpoint, LearningRateMonitor("step")],
        precision=config.train.precision,
        max_steps=config.optim.scheduler.num_training_steps,
        amp_backend=amp_backend,
        gradient_clip_val=config.train.max_grad_norm,
        val_check_interval=min(config.train.validation_interval, 1.0),
        check_val_every_n_epoch=max(int(config.train.validation_interval), 1),
        limit_val_batches=0.0 if config.train.evaluate_after_steps > 0 else 1.0,
        accumulate_grad_batches=config.train.accumulate_grads,
        log_every_n_steps=config.train.logging_interval,
    ).fit(module, datamodule)

    checkpoint_path = (
        checkpoint.best_model_path
        if config.train.save_best_checkpoint
        else checkpoint.last_model_path
    )
    module = FeedbackPrizeModule.load_from_checkpoint(checkpoint_path, config=config)

    module.model.save_pretrained(name)
    datamodule.tokenizer.save_pretrained(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args, unknown_args = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(unknown_args)
    main(config)
