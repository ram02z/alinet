import argparse
from datasets import load_dataset
import datasets
import logging
import os


def get_resolved_subset_file_path(fp, category):
    dir = os.path.dirname(fp)

    filename_with_ext = os.path.basename(fp)
    filename, file_ext = os.path.splitext(filename_with_ext)

    new_fn = f"{filename}-subset-{category}{file_ext}"

    new_fp = os.path.join(dir, new_fn)

    return new_fp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", help="Path to CSV file to split by categories")

    args = parser.parse_args()
    data = load_dataset("csv", data_files=args.file_path, split="train")

    recall_data = data.filter(lambda example: example["category"] == "recall")
    method_data = data.filter(lambda example: example["category"] == "method")
    describe_data = data.filter(lambda example: example["category"] == "description")
    explain_data = data.filter(lambda example: example["category"] == "explanation")

    print("recall:", len(recall_data))
    print("method:", len(method_data))
    print("describe:", len(describe_data))
    print("explain:", len(explain_data))

    PERCENTAGE = 0.10

    recall_prop = int((len(recall_data)) * PERCENTAGE) + 1
    method_prop = int((len(method_data)) * PERCENTAGE) + 1
    describe_prop = int((len(describe_data)) * PERCENTAGE) + 1
    explain_prop = int((len(explain_data)) * PERCENTAGE) + 1

    print("recall_prop:", recall_prop)
    print("method_prop:", method_prop)
    print("describe_prop:", describe_prop)
    print("explain_prop:", explain_prop)

    recall_subset = recall_data.select(range(recall_prop))
    method_subset = method_data.select(range(method_prop))
    describe_subset = describe_data.select(range(describe_prop))
    explain_subset = explain_data.select(range(explain_prop))

    recall_subset.to_csv(get_resolved_subset_file_path(args.file_path, "recall"))
    method_subset.to_csv(get_resolved_subset_file_path(args.file_path, "method"))
    describe_subset.to_csv(get_resolved_subset_file_path(args.file_path, "description"))
    explain_subset.to_csv(get_resolved_subset_file_path(args.file_path, "explanation"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    datasets.logging.set_verbosity_info()
    main()
