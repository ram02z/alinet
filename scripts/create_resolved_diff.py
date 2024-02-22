import argparse
import datasets
from datasets import Dataset, load_dataset
from fastcoref import spacy_component
import spacy
import re
import json
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def create_diff(original, new):
    diff = 0
    new_dataset = []
    for index, info in enumerate(original):
        example = new[index]
        example1 = original[index]
        if example['resolved'] != example1['resolved']:
            example['index'] = index
            new_dataset.append(example)
            diff += 1

    data = Dataset.from_list(new_dataset)
    data.to_csv("data/balanced/train-resolved-diff.csv")

    print(diff)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "original_file_path", help="Path to original CSV file"
    )
    parser.add_argument(
        "new_file_path", help="Path to original CSV file"
    )

    args = parser.parse_args()
    original = load_dataset("csv", data_files=args.original_file_path, split="train")
    new = load_dataset("csv", data_files=args.new_file_path, split="train")

    create_diff(original, new)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    datasets.logging.set_verbosity_info()
    main()