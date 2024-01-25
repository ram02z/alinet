import logging
import os
from dataclasses import dataclass, field

import datasets
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import set_seed, HfArgumentParser
from strenum import StrEnum

logger = logging.getLogger(__name__)


class Dataset(StrEnum):
    BASELINE = "baseline"
    BASELINE_NOISE = "baseline_noise"
    BASELINE_BALANCED = "baseline_balanced"

@dataclass
class GenerateDatasetArguments:
    dataset: Dataset = field(metadata={"help": "Name of the dataset to use"})
    data_dir: str = field(default="data", metadata={"help": "Output data directory"})
    seed: int = field(default=42, metadata={"help": "Random seed"})


def contain_question_mark(data):
    return data["target"][-1].rstrip() == "?"


def normalise(data):
    # Lowercase the text
    data["source"] = data["source"].lower()
    data["target"] = data["target"].lower()

    # Remove new line characters
    data["source"] = data["source"].replace("\n", " ")

    return data

def categorise_dataset(data):
    if any(word in data['target'] for word in ["what"]):
        data['category'] = "description"
    elif any(word in data['target'] for word in ["where", "when", "who", "how many", "how much", "which", "how long"]):
        data['category'] = "recall" 
    elif any(word in data['target'] for word in ["how did", "how does", "how do", "compute", "calculate"]):
        data['category'] = "method"
    elif any(word in data['target'] for word in ["why"]):
        data['category'] = "explanation"
    elif any(word in data['target'] for word in ["compare", "difference"]):
        data['category'] = "comparison" 
    
    return data

def print_distribution(dataset):
    method_ds = dataset.filter(lambda data: data["category"] == "method")
    description_ds = dataset.filter(lambda data: data["category"] == "description")
    explanation_ds = dataset.filter(lambda data: data["category"] == "explanation")
    comparison_ds = dataset.filter(lambda data: data["category"] == "comparison")
    recall_ds = dataset.filter(lambda data: data["category"] == "recall")

    na_ds = dataset.filter(lambda data: data["category"] == "NA")

    print("description distribution =" + str( len(description_ds) / len(dataset) * 100) + "%, count = " + str(len(description_ds)))
    print("recall distribution = " + str( len(recall_ds) / len(dataset) * 100) + "%, count = " + str(len(recall_ds)))
    print("explanation distribution = " + str( len(explanation_ds) / len(dataset) * 100) + "%, count = " + str(len(explanation_ds)))
    print("method distribution = " + str( len(method_ds) / len(dataset) * 100) + "%, count = " + str(len(method_ds)))
    print("comparison distribution = " + str( len(comparison_ds) / len(dataset) * 100) + "%, count = " + str(len(comparison_ds)))
    print("na distribution = " + str( len(na_ds) / len(dataset) * 100) + "%, count = " + str(len(na_ds)))

def remove_na_category(data):
    if data['category'] == 'NA':
        return False
    else:
        return True

def remove_excess_desc_recall(data, countDict):
    if data['category'] == "description" and countDict['desc_count'] < 15970:
        countDict['desc_count'] += 1
        return True
    elif data['category'] == "description" and countDict['desc_count'] >= 15970:
        return False
    elif data['category'] == "recall" and countDict['recall_count'] < 15970:
        countDict['recall_count'] += 1
        return True
    elif data['category'] == "recall" and countDict['recall_count'] >= 15970:
        return False
    else:
        return True
    
def main():
    parser = HfArgumentParser((GenerateDatasetArguments,))
    args = parser.parse_args_into_dataclasses()[0]

    set_seed(args.seed)

    logger.info("loading datasets")
    if args.dataset == Dataset.BASELINE:
        data = (
            load_dataset("squad")
            .select_columns(["context", "question"])
            .rename_columns({"context": "source", "question": "target"})
            .filter(contain_question_mark)
            .map(normalise)
        )
    elif args.dataset == Dataset.BASELINE_NOISE:
        squad_data = (
            load_dataset("squad")
            .select_columns(["context", "question"])
            .rename_columns({"context": "source", "question": "target"})
            .filter(contain_question_mark)
            .map(normalise)
        )
        spoken_squad_data = (
            load_dataset("ram02/spoken_squad")
            .select_columns(["context", "question"])
            .rename_columns({"context": "source", "question": "target"})
            .filter(contain_question_mark)
            .map(normalise)
        )
        train_data = concatenate_datasets(
            [squad_data["train"], spoken_squad_data["train"]]
        )
        valid_data = concatenate_datasets(
            [squad_data["validation"], spoken_squad_data["validation"]]
        )
        data = DatasetDict({"train": train_data, "validation": valid_data})
    elif args.dataset == Dataset.BASELINE_BALANCED:
            squad_data = (
                load_dataset("squad", split="train+validation", trust_remote_code=True)
                .select_columns(["context", "question"])
                .rename_columns({"context": "source", "question": "target"})
            )

            adversarial_data = (
                load_dataset("adversarial_qa", "adversarialQA", split="train+validation+test", trust_remote_code=True)
                .select_columns(["context", "question"])
                .rename_columns({"context": "source", "question": "target"})
            )
            narrative_data = (
                load_dataset("narrativeqa", split="train+validation+test", trust_remote_code=True)
                .select_columns(["document", "question"])
                .map(
                    lambda x: {
                        "document": x["document"]["summary"]["text"],
                        "question": x["question"]["text"],
                    }
                )
                .rename_columns({"document": "source", "question": "target"})
            )
            fairytale_data = (
                load_dataset("GEM/FairytaleQA", split="train+validation+test", trust_remote_code=True)
                .filter(lambda x: x["ex_or_im"] == "explicit")
                .select_columns(["content", "target"])
                .rename_columns({"content": "source"})
            )

            sciq_data = (
                load_dataset("sciq", split="train+validation+test", trust_remote_code=True)
                .select_columns(["support", "question"])
                .rename_columns({"support": "source", "question": "target"})
            )

            dataset = concatenate_datasets(
                [squad_data, adversarial_data, narrative_data, fairytale_data, sciq_data]
            )

            dataset = dataset.filter(contain_question_mark)

            dataset = dataset.map(normalise)

            dataset = dataset.add_column("category", ["NA"] * len(dataset))

            dataset = dataset.map(categorise_dataset)

            comparative_dataset = load_dataset("alinet/comparativeQA", split='train')

            dataset = concatenate_datasets(
                [dataset, comparative_dataset]
            )

            dataset = dataset.filter(remove_na_category)

            countDict = {"desc_count": 0, "recall_count": 0}

            dataset = dataset.filter(remove_excess_desc_recall, fn_kwargs={"countDict": countDict})

            dataset = dataset.class_encode_column("category")

            dataset = dataset.train_test_split(test_size=0.2, stratify_by_column="category")

            data = DatasetDict({"train": dataset["train"], "validation": dataset["test"]})

            


    logger.info("saving dataset")


    data["train"].to_csv(os.path.join(args.data_dir, "train.csv"))
    data["validation"].to_csv(os.path.join(args.data_dir, "validation.csv"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    datasets.logging.set_verbosity_info()
    main()
