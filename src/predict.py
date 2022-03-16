import argparse
import gc
import os
import warnings
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DebertaV2Tokenizer,
)

from dataset import BucketBatchSampler, DataCollatorForNER, NERDataset
from utils import (
    concat_tensors_with_padding,
    convert_deberta_v2_tokenizer,
    convert_offset_by_diffs,
    convert_offsets_to_word_indices,
    extract_entities_from_ner_tags,
    load_articles_with_ids,
    ner_beam_search_decode,
    replace_with_fused_layernorm,
    resolve_encodings_and_normalize,
)

default_minimum_lengths = {
    "Lead": 9,
    "Claim": 3,
    "Position": 5,
    "Rebuttal": 4,
    "Evidence": 14,
    "Counterclaim": 6,
    "Concluding Statement": 11,
}
default_minimum_probs = {
    "Lead": 0.70,
    "Claim": 0.55,
    "Position": 0.55,
    "Rebuttal": 0.55,
    "Evidence": 0.65,
    "Counterclaim": 0.50,
    "Concluding Statement": 0.70,
}

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def update_thresholds(
    args: argparse.Namespace,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    minimum_lengths = default_minimum_lengths.copy()
    minimum_probs = default_minimum_probs.copy()

    for update in args.minimum_lengths:
        name, value = update.split(":")
        minimum_lengths[name.strip()] = int(value.strip())
    for update in args.minimum_probs:
        name, value = update.split(":")
        minimum_probs[name.strip()] = float(value.strip())

    return minimum_lengths, minimum_probs


def prepare_resources(
    args: argparse.Namespace,
) -> Tuple[List[str], List[str], List[str], DataLoader]:
    text_ids, texts = load_articles_with_ids(args.textfile_dir)
    converted_texts = [resolve_encodings_and_normalize(text) for text in texts]

    # Create a tokenizer and change it to the fast version. Note that the original
    # `transformers` does not support `DebertaV2TokenizerFast`, we need to
    # explicitly change it using custom codes.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name[0], add_prefix_space=True)
    if isinstance(tokenizer, DebertaV2Tokenizer):
        tokenizer = convert_deberta_v2_tokenizer(tokenizer)

    # Create a dataloader for the test texts.
    dataloader = DataLoader(
        NERDataset(converted_texts, tokenizer, max_length=args.max_length),
        batch_sampler=BucketBatchSampler(converted_texts, args.batch_size),
        num_workers=os.cpu_count(),
        collate_fn=DataCollatorForNER(tokenizer, pad_to_multiple_of=8),
    )

    # Reorder the texts because bucket batch sampler shuffles the order of samples.
    text_ids = [text_ids[i] for batch in dataloader.batch_sampler for i in batch]
    texts = [texts[i] for batch in dataloader.batch_sampler for i in batch]
    converted_texts = [
        converted_texts[i] for batch in dataloader.batch_sampler for i in batch
    ]
    return text_ids, texts, converted_texts, dataloader


def predict(
    model_name: str, dataloader: DataLoader, args: argparse.Namespace
) -> Tuple[np.ndarray, torch.Tensor, Dict[int, str]]:
    # Load the corresponding token-prediction model.
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    replace_with_fused_layernorm(model)

    model.cuda().eval()
    if args.use_fp16:
        model.half()

    # Predict the BIO-tag token probabilities.
    predictions = []
    for batch in tqdm.tqdm(dataloader):
        batch.pop("texts")
        offset_mapping = batch.pop("offset_mapping")

        batch = {k: v.cuda() for k, v in batch.items()}
        probs = model(**batch).logits.float().softmax(dim=-1)
        predictions.append((offset_mapping, probs))

    # Gather the batchwise predictions to the single tensors.
    offset_mapping, probs = map(list, zip(*predictions))
    return (
        concat_tensors_with_padding(offset_mapping, padding=0).numpy(),
        concat_tensors_with_padding(probs, padding=0),
        model.config.id2label,
    )


@torch.no_grad()
def main(args: argparse.Namespace):
    minimum_lengths, minimum_probs = update_thresholds(args)
    text_ids, texts, converted_texts, dataloader = prepare_resources(args)

    # Collect the BIO-based token predictions for the given models.
    ensemble_probs = 0
    for i, model_name in enumerate(args.model_name):
        offset_mapping, probs, id2label = predict(model_name, dataloader, args)
        ensemble_probs = ensemble_probs + probs / len(args.model_name)

        gc.collect()
        torch.cuda.empty_cache()

    # Generate the NER tags with averaged (ensembled) predictions.
    log_probs = ensemble_probs.log()
    preds, pred_probs = ner_beam_search_decode(log_probs, id2label, args.beam_size)
    preds, pred_probs = preds.cpu().numpy(), pred_probs.cpu().numpy()

    # Extract the entities from the generated tags.
    entities_list = []
    for text, converted_text, preds, pred_probs, offset_mapping in zip(
        texts, converted_texts, preds, pred_probs, offset_mapping
    ):
        valid_mask = offset_mapping[..., 1] > 0
        preds, pred_probs = preds[valid_mask], pred_probs[valid_mask]
        offset_mapping = offset_mapping[valid_mask]

        # Extract the NER entities from BIO-naming tags. Note that the low-confidence or
        # too-short entities will be dropped.
        entities, entity_probs = extract_entities_from_ner_tags(
            [id2label[x] for x in preds], offset_mapping, pred_probs
        )
        entities = convert_offsets_to_word_indices(converted_text, entities)

        entities = [
            (entity, start, end, prob)
            for (entity, start, end), prob in zip(entities, entity_probs)
            if end - start + 1 >= minimum_lengths[entity]
            and prob >= minimum_probs[entity]
        ]
        entities_list.append(entities)

    # Because we change the input text from the original one, the predicted entity spans
    # should be stretched.
    original_entities_list = []
    for text, converted_text, entities in zip(texts, converted_texts, entities_list):
        opcodes = SequenceMatcher(
            None, converted_text.split(), text.split(), False
        ).get_opcodes()

        original_entities = []
        for entity, start, end, confidence in entities:
            original_start = convert_offset_by_diffs(start, opcodes)
            original_end = convert_offset_by_diffs(end, opcodes)
            original_entities.append((entity, original_start, original_end, confidence))
        original_entities_list.append(original_entities)

    # Write the submission file using the extracted entities.
    output = []
    for text_id, entities in zip(text_ids, entities_list):
        for entity, start, end, confidence in entities:
            ps = " ".join(map(str, range(start, end + 1)))
            row = {"id": text_id, "class": entity, "predictionstring": ps}
            if args.return_confidence:
                row["confidence"] = confidence
            output.append(row)
    pd.DataFrame(output).to_csv(args.output_name, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", nargs="+")
    parser.add_argument("--output_name", default="submission.csv")
    parser.add_argument("--textfile_dir", default="./feedback-prize-2021/test")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--use_fp16", action="store_true", default=False)
    parser.add_argument("--minimum_lengths", nargs="+", default=[])
    parser.add_argument("--minimum_probs", nargs="+", default=[])
    parser.add_argument("--return_confidence", action="store_true", default=False)
    main(parser.parse_args())
