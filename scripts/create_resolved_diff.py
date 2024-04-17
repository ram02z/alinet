import argparse
import datasets
from datasets import Dataset, load_dataset
import logging
import os

logger = logging.getLogger(__name__)


def create_diff(original, new):
    diff = 0
    new_dataset = []
    for index in range(len(original)):
        example = new[index]
        example1 = original[index]
        if example["resolved"] != example1["resolved"]:
            example["index"] = index
            new_dataset.append(example)
            diff += 1

    data = Dataset.from_list(new_dataset)
    logger.info(f"number of different resolved: {diff}")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("original_file_path", help="Path to original CSV file")
    parser.add_argument("new_file_path", help="Path to new CSV file")

    args = parser.parse_args()
    original = load_dataset("csv", data_files=args.original_file_path, split="train")
    new = load_dataset("csv", data_files=args.new_file_path, split="train")

    diff_data = create_diff(original, new)

    fp = args.original_file_path
    dir = os.path.dirname(fp)
    filename_with_ext = os.path.basename(fp)
    filename, file_ext = os.path.splitext(filename_with_ext)
    diff_data.to_csv(os.path.join(dir, f"{filename}-diff-{file_ext}"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    datasets.logging.set_verbosity_info()
    main()
