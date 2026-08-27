import json
import os
import uuid

import hydra
import pytorch_lightning as pl
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from tuna.datamodule.ppi_module import PPIDataModule
from tuna.inference.export import save_backbone_from_checkpoint


class EpochSummaryPrinter(Callback):
    @staticmethod
    def _as_float(metrics, key):
        value = metrics.get(key)

        if value is None:
            return None

        try:
            if hasattr(value, "detach"):
                value = value.detach().cpu()

            if hasattr(value, "item"):
                value = value.item()

            return float(value)

        except (TypeError, ValueError):
            return None

    @staticmethod
    def _fmt(value):
        return "NA" if value is None else f"{value:.6f}"

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics

        fields = {
            "train_loss": self._as_float(
                metrics,
                "train/loss",
            ),
            "train_acc": self._as_float(
                metrics,
                "train/accuracy",
            ),
            "train_auprc": self._as_float(
                metrics,
                "train/auprc",
            ),
            "train_auroc": self._as_float(
                metrics,
                "train/auroc",
            ),
            "train_mcc": self._as_float(
                metrics,
                "train/mcc",
            ),
            "val_loss": self._as_float(
                metrics,
                "val/loss",
            ),
            "val_acc": self._as_float(
                metrics,
                "val_ckpt_acc",
            ),
            "val_auprc": self._as_float(
                metrics,
                "val_ckpt_auprc",
            ),
        }

        print(
            "EPOCH_SUMMARY "
            f"epoch={trainer.current_epoch} "
            + " ".join(
                f"{key}={self._fmt(value)}"
                for key, value in fields.items()
            ),
            flush=True,
        )

    def on_test_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        fields = {
            "test_loss": self._as_float(
                metrics,
                "test/loss",
            ),
            "test_acc": self._as_float(
                metrics,
                "test/accuracy",
            ),
            "test_auprc": self._as_float(
                metrics,
                "test/auprc",
            ),
            "test_auroc": self._as_float(
                metrics,
                "test/auroc",
            ),
            "test_mcc": self._as_float(
                metrics,
                "test/mcc",
            ),
            "test_precision": self._as_float(
                metrics,
                "test/precision",
            ),
            "test_recall": self._as_float(
                metrics,
                "test/recall",
            ),
            "test_f1": self._as_float(
                metrics,
                "test/f1",
            ),
        }

        print(
            "TEST_SUMMARY "
            + " ".join(
                f"{key}={self._fmt(value)}"
                for key, value in fields.items()
            ),
            flush=True,
        )


def make_json_serializable(metrics):
    serializable = {}

    for metric_name, value in metrics.items():
        if hasattr(value, "detach"):
            value = value.detach().cpu()

        if hasattr(value, "item"):
            value = value.item()

        try:
            serializable[metric_name] = float(value)
        except (TypeError, ValueError):
            serializable[metric_name] = str(value)

    return serializable


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="config",
)
def main(cfg: DictConfig):
    pl.seed_everything(
        cfg.seed,
        workers=True,
    )

    run_id = str(uuid.uuid4())[:8]

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=run_id,
    )

    checkpoint_dir = os.path.join(
        "checkpoints",
        run_id,
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True,
    )

    lit_module = instantiate(
        cfg.model,
        _recursive_=True,
    )

    last_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="last",
        save_last=True,
        save_top_k=0,
    )

    best_val_acc_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=(
            "best-val-acc-"
            "{epoch:02d}-"
            "{val_ckpt_acc:.5f}"
        ),
        monitor="val_ckpt_acc",
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=False,
    )

    best_val_auprc_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=(
            "best-val-auprc-"
            "{epoch:02d}-"
            "{val_ckpt_auprc:.5f}"
        ),
        monitor="val_ckpt_auprc",
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=False,
    )

    trainer_kwargs = dict(cfg.trainer)

    trainer_kwargs.setdefault(
        "enable_progress_bar",
        False,
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[
            last_checkpoint_callback,
            best_val_acc_checkpoint_callback,
            best_val_auprc_checkpoint_callback,
            EpochSummaryPrinter(),
        ],
        **trainer_kwargs,
    )

    data_module = PPIDataModule(
        config=cfg,
        embedding_type=(
            lit_module
            .model
            .model_backbone
            .embedding_type
        ),
    )

    trainer.fit(
        lit_module,
        data_module,
    )

    checkpoint_paths = {
        "last": (
            last_checkpoint_callback.last_model_path
        ),
        "best_val_accuracy": (
            best_val_acc_checkpoint_callback.best_model_path
        ),
        "best_val_auprc": (
            best_val_auprc_checkpoint_callback.best_model_path
        ),
    }

    missing_checkpoints = [
        name
        for name, path in checkpoint_paths.items()
        if (
            not path
            or not os.path.isfile(path)
        )
    ]

    if missing_checkpoints:
        raise RuntimeError(
            "Missing expected checkpoints: "
            + ", ".join(missing_checkpoints)
        )

    print(
        "\nCHECKPOINTS_SELECTED",
        flush=True,
    )

    for checkpoint_name, checkpoint_path in (
        checkpoint_paths.items()
    ):
        print(
            f"{checkpoint_name}={checkpoint_path}",
            flush=True,
        )

    all_test_results = {}

    for checkpoint_name, checkpoint_path in (
        checkpoint_paths.items()
    ):
        print(
            "\n"
            "========================================\n"
            f"TESTING CHECKPOINT: {checkpoint_name}\n"
            f"PATH: {checkpoint_path}\n"
            "========================================",
            flush=True,
        )

        returned_results = trainer.test(
            model=lit_module,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
            verbose=False,
        )

        raw_metrics = (
            returned_results[0]
            if returned_results
            else {}
        )

        metrics = make_json_serializable(
            raw_metrics
        )

        all_test_results[checkpoint_name] = {
            "checkpoint_path": checkpoint_path,
            "metrics": metrics,
        }

        formatted_metrics = " ".join(
            f"{metric_name}={value:.6f}"
            for metric_name, value in metrics.items()
            if isinstance(value, float)
        )

        print(
            "CHECKPOINT_TEST_SUMMARY "
            f"checkpoint={checkpoint_name} "
            f"{formatted_metrics}",
            flush=True,
        )

        wandb_metrics = {
            (
                "checkpoint_test/"
                f"{checkpoint_name}/"
                f"{metric_name.replace('/', '_')}"
            ): value
            for metric_name, value in metrics.items()
            if isinstance(value, float)
        }

        if wandb_metrics:
            wandb_logger.experiment.log(
                wandb_metrics
            )

    test_results_path = os.path.join(
        checkpoint_dir,
        "test_results_all_checkpoints.json",
    )

    with open(
        test_results_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            all_test_results,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    print(
        "\nAll checkpoint test results written to:",
        test_results_path,
        flush=True,
    )

    if cfg.save_model.save:
        save_root = os.path.join(
            cfg.save_model.save_dir,
            run_id,
        )

        export_names = {
            "last": "last",
            "best_val_accuracy": "best_val_acc",
            "best_val_auprc": "best_val_auprc",
        }

        for checkpoint_name, checkpoint_path in (
            checkpoint_paths.items()
        ):
            save_backbone_from_checkpoint(
                checkpoint_path,
                cfg.model.model_backbone,
                os.path.join(
                    save_root,
                    export_names[checkpoint_name],
                ),
            )

    wandb.finish()


if __name__ == "__main__":
    main()
