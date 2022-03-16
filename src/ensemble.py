import argparse
from typing import List, Tuple

import pandas as pd
import tqdm

from utils import group_overlapped_entities


def create_entities_from_prediction(pred: pd.DataFrame) -> List[Tuple[str, int, int]]:
    starts = pred["predictionstring"].str.split().str[0].astype(int)
    ends = pred["predictionstring"].str.split().str[-1].astype(int)
    return list(zip(pred["class"], starts, ends, pred["confidence"]))


def main(args: argparse.Namespace):
    submissions = [
        dict(list(pd.read_csv(submission_name).groupby("id")))
        for submission_name in args.submission
    ]

    output = []
    for text_id in tqdm.tqdm(submissions[0]):
        predictions = [submission.get(text_id, None) for submission in submissions]
        entities_list = [
            create_entities_from_prediction(x) if x is not None else []
            for x in predictions
        ]
        entity_groups = group_overlapped_entities(sum(entities_list, []))

        for group in entity_groups:
            if len(group) < args.min_threshold:
                continue

            # It is important to select the way to decide the entity length from the
            # grouped entities. We support `union`, `intersect`, `longest`, `shortest`
            # and `mean` entity length types.
            if args.group_strategy == "union":
                start, end = min(x[1] for x in group), max(x[2] for x in group)
            elif args.group_strategy == "intersect":
                start, end = max(x[1] for x in group), min(x[2] for x in group)
            elif args.group_strategy == "longest":
                start, end, _ = max(
                    ((x[1], x[2], x[2] - x[1] + 1) for x in group), key=lambda i: i[2]
                )
            elif args.group_strategy == "shortest":
                start, end, _ = min(
                    ((x[1], x[2], x[2] - x[1] + 1) for x in group), key=lambda i: i[2]
                )
            elif args.group_strategy == "mean":
                start = int(sum(x[1] for x in group) / len(group))
                end = int(sum(x[2] for x in group) / len(group))
            else:
                raise NotImplementedError(f"{args.group_strategy} is not supported.")

            if end < start:
                continue

            ps = " ".join(map(str, range(start, end + 1)))
            output.append({"id": text_id, "class": group[0][0], "predictionstring": ps})

    pd.DataFrame(output).to_csv(args.output_name, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", nargs="+")
    parser.add_argument("--output_name", default="submission.csv")
    parser.add_argument("--min_threshold", type=int, default=2)
    parser.add_argument("--group_strategy", default="mean")
    main(parser.parse_args())
