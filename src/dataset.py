from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy

from utils import generate_ner_tags_from_entities


@dataclass
class NERDataset(Dataset):
    texts: List[str]
    tokenizer: PreTrainedTokenizerBase
    entities_list: Optional[List[List[Tuple[str, int, int]]]] = None
    label2id: Optional[Dict[str, int]] = None
    max_length: int = 2048

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        encoding = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        outputs = dict(**encoding, texts=self.texts[index])

        if self.entities_list is not None:
            ner_tags = generate_ner_tags_from_entities(
                self.entities_list[index], encoding.offset_mapping
            )
            outputs["labels"] = [self.label2id[x] for x in ner_tags]
        return outputs


class BucketBatchSampler(Sampler[Iterable[int]]):
    """A batch sampler for sequence bucketing.

    This class creates buckets according to the length of examples. It first sorts the
    lengths and creates index map. Then it groups them into buckets and shuffle
    randomly. This makes each batch has examples of which lengths are almost same. It
    leads the decrement of unnecessary and wasted paddings, hence, you can reduce the
    padded sequence lengths and entire computational costs.

    Args:
        texts: A list of target texts.
        batch_size: The number of examples in each batch.
    """

    def __init__(self, texts: List[str], batch_size: int):
        indices = np.argsort([len(text.split()) for text in texts])
        if len(indices) % batch_size > 0:
            padding = batch_size - len(indices) % batch_size
            indices = np.append(indices, [-1] * padding)

        self.buckets = indices.reshape(-1, batch_size)
        self.permutation = np.random.permutation(self.buckets.shape[0])

    def __len__(self) -> int:
        return self.buckets.shape[0]

    def __iter__(self) -> Iterator[Iterable[int]]:
        for indices in self.buckets[self.permutation]:
            yield indices[indices >= 0]


@dataclass
class DataCollatorForNER:
    """A collator for NER batches.

    While `NERDataset` returns not only the encoded results but also `texts`, `labels`
    and `offset_mapping`, this class performs additional collatings such as gathering
    and padding.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        total_length = len(batch["input_ids"][0])

        if "offset_mapping" in batch:
            batch["offset_mapping"] = self.pad_sequences(
                batch["offset_mapping"], total_length, (0, 0)
            )
        if "labels" in batch:
            batch["labels"] = self.pad_sequences(batch["labels"], total_length, -100)

        return {
            k: torch.tensor(v, dtype=torch.int64) if not isinstance(v[0], str) else v
            for k, v in batch.items()
        }

    def pad_sequences(
        self, sequences: List[List[int]], total_length: int, padding_value: Any = -1
    ) -> List[List[int]]:
        """Pad the sequences with corresponding padding values.

        Args:
            sequences: The list of sequences. They should have different length.
            total_length: The target length that the sequences should have.
            padding_value: The padding value which will be used to fill to the padded
                sequences. Default is `-1`.

        returns:
            A list of padded sequences.
        """
        if self.tokenizer.padding_side == "right":
            return [x + [padding_value] * (total_length - len(x)) for x in sequences]
        else:
            return [[padding_value] * (total_length - len(x)) + x for x in sequences]
