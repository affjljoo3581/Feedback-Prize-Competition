from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def generate_ner_tags_from_entities(
    entities: List[Tuple[str, int, int]], offset_mapping: List[Tuple[int, int]]
) -> List[str]:
    """Generate NER-tags (with BIO naming) for subword tokens from the entities.

    Args:
        entities: The list of entities which consist of an entity name with its offset
            mappings.
        offset_mapping: The list of offsets which are positions of the tokens.

    Returns:
        A list of NER-tags encoded from the given entity informations.
    """
    ner_tags = ["O" for _ in offset_mapping]
    for entity, entity_start, entity_end in sorted(entities, key=lambda x: x[1]):
        current_ner_tag = f"B-{entity}"
        for i, (token_start, token_end) in enumerate(offset_mapping):
            if min(entity_end, token_end) - max(entity_start, token_start) > 0:
                ner_tags[i] = current_ner_tag
                current_ner_tag = f"I-{entity}"
    return ner_tags


def extract_entities_from_ner_tags(
    ner_tags: List[str], offset_mapping: np.ndarray, probs: Optional[np.ndarray] = None
) -> Tuple[List[Tuple[str, int, int]], List[float]]:
    """Extract the entities from NER-tagged subword tokens.

    This function detects the entities from BIO NER-tags and collects them with
    averaging their confidences (prediction probabilities). Using the averaged
    probabilities, you can filter the low-confidence entities.

    Args:
        ner_tags: The list of subword-token-level NER-tags.
        offset_mapping: The list of offsets which are positions of the tokens.
        probs: An optional prediction probabilities of the subword tokens. Default is
            `None`.

    Returns:
        A tuple of collected NER entities with their averaged entity confidencs
        (prediction probabilities).
    """
    probs = probs if probs is not None else np.zeros(offset_mapping.shape[0])

    entities, gathered_probs, entity, i = [], [], None, None
    for j, ner_tag in enumerate(ner_tags):
        if entity is not None and ner_tag != f"I-{entity}":
            entities.append((entity, offset_mapping[i][0], offset_mapping[j - 1][1]))
            gathered_probs.append(probs[i:j].mean())
            entity = None
        if ner_tag.startswith("B-"):
            entity, i = ner_tag[2:], j

    # Because BIO-naming does not ensure the end of the entities (i.e. E-tag), we cannot
    # automatically detect the end of the last entity in the above loop.
    if entity is not None:
        entities.append((entity, offset_mapping[i][0], offset_mapping[-1][1]))
        gathered_probs.append(probs[i:].mean())
    return entities, gathered_probs


def convert_offsets_to_word_indices(
    text: str, entities: List[Tuple[str, int, int]]
) -> List[Tuple[str, int, int]]:
    """Convert the offset-based spans for the entities to word-level mapping.

    Args:
        text: The original text string.
        entities: The list of entities which consist of an entity name with its offset
            mappings.

    Returns:
        A list of converted entities which have word-level mapping spans.
    """
    word_level_entities, total_num_words = [], len(text.split())
    for entity, start, end in entities:
        start_word_index = len(text[:start].split())
        span_length = len(text[start:end].split())
        end_word_index = min(start_word_index + span_length, total_num_words) - 1
        word_level_entities.append((entity, start_word_index, end_word_index))
    return word_level_entities


def group_overlapped_entities(
    entities: List[Tuple[str, int, int, float]]
) -> List[List[Tuple[str, int, int]]]:
    """Group the similar entities from various predictions.

    Args:
        entities: The list of entities from same target with their confidence. It is
            possible that there are same entities in the list.

    Returns:
        A group of similar entities.
    """
    entity_groups = []
    for entity, ea, eb, confidence in sorted(entities, key=lambda x: x[3]):
        is_grouped = False

        # Find the most similar group and add the entity to the group.
        for group in entity_groups:
            ga, gb = np.mean([x[1] for x in group]), np.mean([x[2] for x in group])
            overlaps = max(min(gb, eb) - max(ga, ea) + 1, 0)
            overlaps = (overlaps / (gb - ga + 1), overlaps / (eb - ea + 1))
            if entity == group[0][0] and max(overlaps) >= 0.5:
                group.append((entity, ea, eb, confidence))
                is_grouped = True
                break

        # If there is no proper group, create new entity group.
        if not is_grouped:
            entity_groups.append([(entity, ea, eb, confidence)])
    return entity_groups


def calculate_ner_entity_matches(
    pred_entities: List[Tuple[str, int, int]],
    label_entities: List[Tuple[str, int, int]],
) -> Tuple[List[str], List[str]]:
    """Calculate the matches between prediction and label.

    This function calculates the overlap ratios between prediction entities and label
    entities. After that, the max-overlapped (best-matched) entities will be collected.
    Finally, the matched and non-matched entities will be returned. You can use this
    matches to calculate an entity-level F1-score.

    Args:
        pred_entities: The list of word-level predicted entities which consist of an
            entity name with its word index range.
        label_entities: The list of word-level ground-truth entities which consist of an
            entity name with its word index range.

    Returns:
        A tuple of lists of matched and non-matched entities.
    """
    candidate_pairs = []
    for i, (pred, pa, pb) in enumerate(pred_entities):
        for j, (label, la, lb) in enumerate(label_entities):
            if pred == label:
                overlaps = max(min(pb, lb) - max(pa, la) + 1, 0)
                overlaps = (overlaps / (pb - pa + 1), overlaps / (lb - la + 1))
                if min(overlaps) >= 0.5:
                    # All overlap ratios should be greater than `0.5`.
                    candidate_pairs.append((pred, max(overlaps), i, j))

    matched_entities = []
    while candidate_pairs:
        # Note that we will only use the max-overlapped matches. So we need to filter
        # the best-matched entities from the prediction and label.
        entity, _, i, j = max(candidate_pairs, key=lambda x: x[1])
        candidate_pairs = [x for x in candidate_pairs if x[2] != i and x[3] != j]
        matched_entities.append(entity)

    total_entities = [x[0] for x in pred_entities] + [x[0] for x in label_entities]
    return matched_entities, total_entities


def ner_entity_macro_f1_score(
    pred_entities_list: List[List[Tuple[str, int, int]]],
    label_entities_list: List[List[Tuple[str, int, int]]],
    entity_types: Optional[List[str]] = None,
) -> float:
    """Calculate macro NER-entitiy-level F1 score.

    Args:
        pred_entities_list: The list of list of word-level predicted entities which
            consist of an entity name with its word index range.
        label_entities_list: The list of list of word-level ground-truth entities which
            consist of an entity name with its word index range.
        entity_types: An optional list of total entity types. If not given, the entity
            types are automatically detected from either predictions and labels. Default
            is `None`.

    Return:
        A entity-level macro-F1 score.
    """
    matched_entities, total_entities = [], []
    for pred_entities, label_entities in zip(pred_entities_list, label_entities_list):
        matched, total = calculate_ner_entity_matches(pred_entities, label_entities)
        matched_entities += matched
        total_entities += total

    # The original calculation of f1-score is `tp / (tp + 0.5 * (fp + fn))`.
    # Interestingly, we know that **true-positive + false-positive = prediction** and
    # **true-positive + false-negative = ground-truth**. Hence we obtain that f1-score
    # is `2 * tp / (tpfp + tpfn) = 2 * tp / (preds + labels)`. Using this formulation,
    # we will count the matched and non-matched entities.
    tp, tpfptpfn = Counter(matched_entities), Counter(total_entities)

    if entity_types is None:
        entity_types = list({*tp, *tpfptpfn})
    return np.mean([2 * tp[k] / (tpfptpfn[k] + 1e-10) for k in entity_types])


def create_ner_conditional_masks(id2label: Dict[int, str]) -> torch.Tensor:
    """Create a NER-conditional mask matrix which implies the relations between
    before-tag and after-tag.

    According to the rule of BIO-naming system, it is impossible that `I-Dog` cannot be
    appeard after `B-Dog` or `I-Dog` tags. This function creates the calculable
    relation-based conditional matrix to prevent from generating wrong tags.

    Args:
        id2label: A dictionary which maps class indices to their label names.

    Returns:
        A conditional mask tensor.
    """
    conditional_masks = torch.zeros(len(id2label), len(id2label))
    for i, before in id2label.items():
        for j, after in id2label.items():
            if after == "O" or after.startswith("B-") or after == f"I-{before[2:]}":
                conditional_masks[i, j] = 1.0
    return conditional_masks


def ner_beam_search_decode(
    log_probs: torch.Tensor, id2label: Dict[int, str], beam_size: int = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode NER-tags from the predicted log-probabilities using beam-search.

    This function decodes the predictions using beam-search algorithm. Because all tags
    are predicted simultaneously while the tags have dependencies of their previous
    tags, the greedy algorithm cannot decode the tags properly. With beam-search, it is
    possible to prevent the below situation:

        >>> sorted = probs[t].sort(dim=-1)
        >>> print("\t".join([f"{id2label[i]} {p}" for p, i in zip()]))
        I-Dog 0.54  B-Cat 0.44  ...
        >>> sorted = probs[t + 1].sort(dim=-1)
        >>> print("\t".join([f"{id2label[i]} {p}" for p, i in zip()]))
        I-Cat 0.99  I-Dog 0.01  ...

    The above shows that if the locally-highest tags are selected, then `I-Dog, I-Dog`
    will be generated even the confidence of the second tag `I-Dog` is significantly
    lower than `I-Cat`. It is more natural that `B-Cat, I-Cat` is generated rather than
    `I-Dog, I-Dog`. The beam-search for NER-tagging task can solve this problem.

    Args:
        log_probs: The log-probabilities of the token predictions.
        id2label: A dictionary which maps class indices to their label names.
        beam_size: The number of candidates for each search step. Default is `2`.

    Returns:
        A tuple of beam-searched indices and their probability tensors.
    """
    # Create the log-probability mask for the invalid predictions.
    log_prob_masks = -10000.0 * (1 - create_ner_conditional_masks(id2label))
    log_prob_masks = log_prob_masks.to(log_probs.device)

    beam_search_shape = (log_probs.size(0), beam_size, log_probs.size(1))
    searched_tokens = log_probs.new_zeros(beam_search_shape, dtype=torch.long)
    searched_log_probs = log_probs.new_zeros(beam_search_shape)

    searched_scores = log_probs.new_zeros(log_probs.size(0), beam_size)
    searched_scores[:, 1:] = -10000.0

    for i in range(log_probs.size(1)):
        # Calculate the accumulated score (log-probabilities) with excluding invalid
        # next-tag predictions.
        scores = searched_scores.unsqueeze(2)
        scores = scores + log_probs[:, i, :].unsqueeze(1)
        scores = scores + (log_prob_masks[searched_tokens[:, :, i - 1]] if i > 0 else 0)

        # Select the top-k (beam-search size) predictions.
        best_scores, best_indices = scores.flatten(1).topk(beam_size)
        best_tokens = best_indices % scores.size(2)
        best_log_probs = log_probs[:, i, :].gather(dim=1, index=best_tokens)

        best_buckets = best_indices.div(scores.size(2), rounding_mode="floor")
        best_buckets = best_buckets.unsqueeze(2).expand(-1, -1, log_probs.size(1))

        # Gather the best buckets and their log-probabilities.
        searched_tokens = searched_tokens.gather(dim=1, index=best_buckets)
        searched_log_probs = searched_log_probs.gather(dim=1, index=best_buckets)

        # Update the predictions by inserting to the corresponding timestep.
        searched_scores = best_scores
        searched_tokens[:, :, i] = best_tokens
        searched_log_probs[:, :, i] = best_log_probs

    # Return the best beam-searched sequence and its probabilities.
    return searched_tokens[:, 0, :], searched_log_probs[:, 0, :].exp()
