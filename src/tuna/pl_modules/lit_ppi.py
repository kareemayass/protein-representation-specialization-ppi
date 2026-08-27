import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score

from tuna.models._mlp import MLP
from tuna.models._transformer import Transformer
from tuna.models.llgp_utils import LLGPMode
from tuna.models.ppi_predictor import PPIPredictor
from tuna.pl_modules.base_module import BaseModule


class LitPPI(BaseModule):
    def __init__(self, model_backbone: Transformer | MLP, optimizer_config: dict):
        super().__init__(config=OmegaConf.create(optimizer_config))
        self.model = PPIPredictor(model_backbone=model_backbone)
        self._initialize_weights()
        self.criterion = nn.BCEWithLogitsLoss()
        self.save_hyperparameters()

        self._val_ckpt_probs = []
        self._val_ckpt_y = []

    def _initialize_weights(self):
        for p in self.model.model_backbone.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def on_validation_epoch_start(self):
        self._val_ckpt_probs = []
        self._val_ckpt_y = []

    def on_validation_epoch_end(self):
        if not self._val_ckpt_y:
            return

        y = torch.cat([t.reshape(-1) for t in self._val_ckpt_y]).int()
        probs = torch.cat([t.reshape(-1) for t in self._val_ckpt_probs]).float()

        preds = (probs >= 0.5).int()
        acc = (preds == y).float().mean()

        try:
            auprc_value = average_precision_score(
                y.detach().cpu().numpy(),
                probs.detach().cpu().numpy(),
            )
        except ValueError:
            auprc_value = float("nan")

        auprc = torch.tensor(float(auprc_value), device=self.device)

        self.log(
            "val_ckpt_acc",
            acc.to(self.device),
            prog_bar=True,
            logger=True,
            sync_dist=False,
        )
        self.log(
            "val_ckpt_auprc",
            auprc,
            prog_bar=True,
            logger=True,
            sync_dist=False,
        )

    def _shared_step(self, batch, mode: LLGPMode, prefix: str):
        if len(batch) == 3:
            proteinA, proteinB, y = batch
            proteinA_lens = None
            proteinB_lens = None
        elif len(batch) == 5:
            proteinA, proteinB, y, proteinA_lens, proteinB_lens = batch
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} items")

        logits = self.model(
            proteinA=proteinA,
            proteinB=proteinB,
            mode=mode,
            lengthsA=proteinA_lens,
            lengthsB=proteinB_lens,
            is_last_epoch=self._is_last_epoch(),
        )

        probs, preds = self._process_logits(logits)
        loss = self.criterion(logits, y.float())

        self.log(
            f"{prefix}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(proteinA),
        )

        self._update_metrics(y, preds, probs, stage=prefix)

        if prefix == "val":
            self._val_ckpt_probs.append(probs.detach().cpu())
            self._val_ckpt_y.append(y.detach().cpu())

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, LLGPMode.TRAINING, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, LLGPMode.VALIDATION, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, LLGPMode.INFERENCE, "test")
