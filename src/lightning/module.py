from typing import Any, Dict, List, Tuple

import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.optim import Optimizer
from transformers import AutoModelForTokenClassification, get_scheduler

from utils import (
    concat_tensors_with_padding,
    convert_offsets_to_word_indices,
    extract_entities_from_ner_tags,
    get_parameter_groups,
    ner_beam_search_decode,
    ner_entity_macro_f1_score,
    reinit_last_layers,
    replace_with_fused_layernorm,
)

try:
    from apex.optimizers import FusedAdam as AdamW
except ModuleNotFoundError:
    from torch.optim import AdamW


class FeedbackPrizeModule(LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.minimum_lengths = dict(config.model.decoding.minimum_lengths)
        self.minimum_probs = dict(config.model.decoding.minimum_probs)

        # Create labels for the NER BIO-naming tags.
        labels = sorted(self.config.model.decoding.minimum_lengths)
        labels = ["O"] + [f"B-{x}" for x in labels] + [f"I-{x}" for x in labels]
        self.id2label = {i: label for i, label in enumerate(labels)}

        torch.manual_seed(config.model.random_seed)
        self.model = AutoModelForTokenClassification.from_pretrained(
            **config.model.transformer,
            id2label=self.id2label,
            label2id={v: k for v, k in self.id2label.items()},
        )

        # Enable gradient checkpointing if the configuration is set.
        if config.train.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        replace_with_fused_layernorm(self)
        reinit_last_layers(self.model, config.model.num_reinit_layers)

    def training_step(self, batch: Dict[str, torch.Tensor], idx: int) -> torch.Tensor:
        [batch.pop(key) for key in ["texts", "offset_mapping"]]
        loss = self.model(**batch).loss
        self.log("train/loss", loss)
        return loss

    def training_step_end(self, outputs: List[torch.Tensor]):
        if (
            self.global_step > self.config.train.evaluate_after_steps
            and self.trainer.limit_val_batches == 0.0
        ):
            self.trainer.limit_val_batches = 1.0
            self.trainer.reset_val_dataloader()

    def validation_step(
        self, batch: Dict[str, torch.Tensor], idx: int
    ) -> Tuple[torch.Tensor, ...]:
        texts, offset_mapping = [batch.pop(key) for key in ["texts", "offset_mapping"]]
        output = self.model(**batch)
        return output.loss, texts, output.logits, batch["labels"], offset_mapping

    def validation_epoch_end(self, outputs: List[Tuple[torch.Tensor, ...]]):
        loss, texts, logits, labels, offset_mapping = map(list, zip(*outputs))
        texts = sum(texts, [])

        # Decode the NER-tags with beam-search algorithm.
        preds, pred_probs = ner_beam_search_decode(
            concat_tensors_with_padding(logits, padding=0).float().log_softmax(dim=-1),
            self.id2label,
            self.config.model.decoding.beam_size,
        )
        labels = concat_tensors_with_padding(labels, padding=-100)
        offset_mapping = concat_tensors_with_padding(offset_mapping, padding=0)

        preds, pred_probs = preds.cpu().numpy(), pred_probs.cpu().numpy()
        labels, offset_mapping = labels.cpu().numpy(), offset_mapping.cpu().numpy()

        # Collect the NER entities for predictions and labels to calculate the F1 score.
        pred_entities_list, label_entities_list = [], []
        for text, preds, pred_probs, labels, offset_mapping in zip(
            texts, preds, pred_probs, labels, offset_mapping
        ):
            valid_mask = offset_mapping[..., 1] > 0

            preds, pred_probs = preds[valid_mask], pred_probs[valid_mask]
            labels, offset_mapping = labels[valid_mask], offset_mapping[valid_mask]

            # Extract the NER entities from BIO-naming tags. Note that the
            # low-confidence or too-short entities will be dropped.
            pred_entities, pred_entity_probs = extract_entities_from_ner_tags(
                [self.id2label[x] for x in preds], offset_mapping, pred_probs
            )
            pred_entities = convert_offsets_to_word_indices(text, pred_entities)

            pred_entities = [
                (entity, a, b)
                for (entity, a, b), prob in zip(pred_entities, pred_entity_probs)
                if b - a + 1 >= self.minimum_lengths[entity]
                and prob >= self.minimum_probs[entity]
            ]
            pred_entities_list.append(pred_entities)

            # Of course, we will extract the entities for labels.
            label_entities, _ = extract_entities_from_ner_tags(
                [self.id2label[x] for x in labels], offset_mapping
            )
            label_entities = convert_offsets_to_word_indices(text, label_entities)
            label_entities_list.append(label_entities)

        # Calculate the macro-F1 score for NER entities.
        f1 = ner_entity_macro_f1_score(pred_entities_list, label_entities_list)

        self.log("val/loss", torch.stack(loss).mean())
        self.log("val/f1_score", f1)

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[Dict[str, Any]]]:
        optimizer = AdamW(get_parameter_groups(self), **self.config.optim.optimizer)
        scheduler = get_scheduler(optimizer=optimizer, **self.config.optim.scheduler)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        if "amp_scaling_state" in checkpoint:
            checkpoint.pop("amp_scaling_state")
