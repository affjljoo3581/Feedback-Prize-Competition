import codecs
import os
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from text_unidecode import unidecode


def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start : error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start : error.end].decode("cp1252"), error.end


# Register the encoding and decoding error handlers for `utf-8` and `cp1252`.
codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)


def load_articles_with_ids(
    textfile_dir: str, sort_by_name: bool = True
) -> Tuple[List[str], List[str]]:
    """Load all articles and collect their ids.

    Args:
        textfile_dir: The directory containing the article files.
        sort_by_name: If enabled, the articles will be sorted by their ids.

    Returns:
        A tuple of a list of text-ids and a list of texts.
    """
    filenames = os.listdir(textfile_dir)
    if sort_by_name:
        filenames = sorted(filenames)

    text_ids, texts = [], []
    for filename in filenames:
        with open(os.path.join(textfile_dir, filename)) as fp:
            text_ids.append(os.path.splitext(filename)[0])
            texts.append(fp.read())
    return text_ids, texts


def create_discourse_entities(
    dataset: pd.DataFrame,
) -> Dict[str, List[List[Tuple[str, int, int]]]]:
    """Create discourse entities from the given dataset.

    The discourse dataset consists of separated discourse informations. This function
    extracts the discourse entities from the dataset and groups by their text-ids. Using
    this function, you can easily create NER entities and convert to BIO-naming tags.

    Args:
        dataset: The pandas dataframe containing the discourse informations.

    Returns:
        A dictionary for entity informations (with offset-mapping based spans).
    """
    entities_dict = {}
    for text_id, discourses in dict(list(dataset.groupby("id"))).items():
        entities = discourses.discourse_type
        starts = discourses.discourse_start.astype(int)
        ends = discourses.discourse_end.astype(int)
        entities_dict[text_id] = list(zip(entities, starts, ends))
    return entities_dict


def resolve_encodings_and_normalize(text: str) -> str:
    """Resolve the encoding problems and normalize the abnormal characters."""
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text


def convert_offset_by_diffs(
    offset: int, opcodes: List[Tuple[str, int, int, int, int]]
) -> int:
    """Convert the offset of text according to the opcodes from diff-comparison.

    Args:
        offset: The target offset of the text.
        opcodes: The output of `difflib.SequenceMatcher.get_opcodes` method.

    Returns:
        A converted offset of the text.
    """
    new_offset = 0
    for _, src_a, src_b, dst_a, dst_b in opcodes:
        if offset <= src_a:
            break
        if offset > src_b:
            new_offset += dst_b - dst_a
        elif src_b - src_a == dst_b - dst_a:
            new_offset += offset - src_a
        else:
            new_offset += (offset - src_a) / (src_b - src_a) * (dst_b - dst_a)
    return int(new_offset)


def split_ner_stratified_kfold(
    entities_list: List[List[Tuple[str, int, int]]], num_folds: int, fold_index: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Split to the NER entity-level stratified k-folds.

    Args:
        entities_list: The list of list of entities.
        num_folds: The number of folds to split.
        fold_index: The index of current fold.

    Returns:
        A tuple of indices for train and validation.
    """
    # Collect the entity types and sort them for deterministics.
    entity_types = sorted({y for x in entities_list for y, *_ in x})

    # Count the entity appearances and transform to vectors for the multilabel k-fold.
    entity_counts = [Counter(y for y, *_ in x) for x in entities_list]
    entity_labels = [[cnt[x] for x in entity_types] for cnt in entity_counts]

    kfold = MultilabelStratifiedKFold(num_folds, shuffle=True, random_state=42)
    return list(kfold.split(entities_list, entity_labels))[fold_index]
