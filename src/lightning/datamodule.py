import os
from difflib import SequenceMatcher
from typing import Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DebertaV2Tokenizer

from dataset import BucketBatchSampler, DataCollatorForNER, NERDataset
from utils import (
    convert_deberta_v2_tokenizer,
    convert_offset_by_diffs,
    create_discourse_entities,
    load_articles_with_ids,
    resolve_encodings_and_normalize,
    split_ner_stratified_kfold,
)


class FeedbackPrizeDataModule(LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.num_dataloader_workers = (
            self.config.dataset.dataloader_workers
            if self.config.dataset.dataloader_workers >= 0
            else os.cpu_count()
        )

    def setup(self, stage: Optional[str] = None):
        text_ids, texts = load_articles_with_ids(self.config.dataset.textfile_dir)
        labels = sorted(self.config.model.decoding.minimum_lengths)
        labels = ["O"] + [f"B-{x}" for x in labels] + [f"I-{x}" for x in labels]

        entities_dict = create_discourse_entities(
            pd.read_csv(self.config.dataset.dataset_filename)
        )
        entities_list = [entities_dict[text_id] for text_id in text_ids]

        # Convert the texts and change the offsets of the discourse spans.
        if self.config.dataset.normalize:
            converted_texts, converted_entities_list = [], []
            for text, entities in zip(texts, entities_list):
                converted_text = resolve_encodings_and_normalize(text)
                if len(text) == len(converted_text):
                    converted_texts.append(converted_text)
                    converted_entities_list.append(entities)
                    continue

                converted_entities = []
                ops = SequenceMatcher(None, text, converted_text, False).get_opcodes()
                for entity, start, end in entities:
                    converted_start = convert_offset_by_diffs(start, ops)
                    converted_end = convert_offset_by_diffs(end, ops)
                    converted_entities.append((entity, converted_start, converted_end))

                converted_texts.append(converted_text)
                converted_entities_list.append(converted_entities)
        else:
            converted_texts, converted_entities_list = texts, entities_list

        # Split the texts into train and validation using stratified k-fold split.
        train_indices, val_indices = split_ner_stratified_kfold(
            converted_entities_list,
            self.config.dataset.num_folds,
            self.config.dataset.fold_index,
        )

        # Create a tokenizer and change it to the fast version. Note that the original
        # `transformers` does not support `DebertaV2TokenizerFast`, we need to
        # explicitly change it using custom codes.
        self.tokenizer = AutoTokenizer.from_pretrained(
            **self.config.model.transformer, add_prefix_space=True
        )
        if isinstance(self.tokenizer, DebertaV2Tokenizer):
            self.tokenizer = convert_deberta_v2_tokenizer(self.tokenizer)

        # Create datasets for train and validation.
        self.train_dataset = NERDataset(
            texts=[converted_texts[i] for i in train_indices],
            tokenizer=self.tokenizer,
            entities_list=[converted_entities_list[i] for i in train_indices],
            label2id={label: i for i, label in enumerate(labels)},
            max_length=self.config.dataset.max_length,
        )
        self.val_dataset = NERDataset(
            texts=[converted_texts[i] for i in val_indices],
            tokenizer=self.tokenizer,
            entities_list=[converted_entities_list[i] for i in val_indices],
            label2id={label: i for i, label in enumerate(labels)},
            max_length=self.config.dataset.max_length,
        )

        # Create sequence-bucket batch samplers for train and validation.
        np.random.seed(self.config.dataset.random_seed)
        self.train_batch_sampler = BucketBatchSampler(
            self.train_dataset.texts, self.config.train.batch_size
        )
        self.val_batch_sampler = BucketBatchSampler(
            self.val_dataset.texts, self.config.train.batch_size
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            num_workers=self.num_dataloader_workers,
            collate_fn=DataCollatorForNER(self.tokenizer, pad_to_multiple_of=8),
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_sampler=self.val_batch_sampler,
            num_workers=self.num_dataloader_workers,
            collate_fn=DataCollatorForNER(self.tokenizer, pad_to_multiple_of=8),
            persistent_workers=True,
        )
