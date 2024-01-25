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
    # to prevent re-categorising comparisonQA data
    if data["category"] == "comparison":
        return data


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

def remove_excess_desc_recall_train(data, countDict):
    if data['category'] == "description" and countDict['desc_count'] < 12558:
        countDict['desc_count'] += 1
        return True
    elif data['category'] == "description" and countDict['desc_count'] >= 12558:
        return False
    elif data['category'] == "recall" and countDict['recall_count'] < 12558:
        countDict['recall_count'] += 1
        return True
    elif data['category'] == "recall" and countDict['recall_count'] >= 12558:
        return False
    else:
        return True
    
def remove_excess_desc_recall_validate(data, countDict):
    if data['category'] == "description" and countDict['desc_count'] < 3413:
        countDict['desc_count'] += 1
        return True
    elif data['category'] == "description" and countDict['desc_count'] >= 3413:
        return False
    elif data['category'] == "recall" and countDict['recall_count'] < 3413:
        countDict['recall_count'] += 1
        return True
    elif data['category'] == "recall" and countDict['recall_count'] >= 3413:
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
                load_dataset("squad", trust_remote_code=True)
                .select_columns(["context", "question"])
                .rename_columns({"context": "source", "question": "target"})
            )

            adversarial_data = (
                load_dataset("adversarial_qa", "adversarialQA", trust_remote_code=True)
                .select_columns(["context", "question"])
                .rename_columns({"context": "source", "question": "target"})
            )
            narrative_data = (
                load_dataset("narrativeqa", trust_remote_code=True)
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
                load_dataset("GEM/FairytaleQA", trust_remote_code=True)
                .filter(lambda x: x["ex_or_im"] == "explicit")
                .select_columns(["content", "target"])
                .rename_columns({"content": "source"})
            )

            sciq_data = (
                load_dataset("sciq", trust_remote_code=True)
                .select_columns(["support", "question"])
                .rename_columns({"support": "source", "question": "target"})
            )

            train_dataset = concatenate_datasets(
            [squad_data["train"], adversarial_data["train"], narrative_data["train"], fairytale_data["train"], sciq_data["train"]]
            )

            validate_dataset = concatenate_datasets(
            [squad_data["validation"], adversarial_data["validation"], adversarial_data["test"], narrative_data["validation"], narrative_data["test"], fairytale_data["validation"], fairytale_data["test"], sciq_data["validation"], sciq_data["test"]]
            )

            train_dataset = train_dataset.filter(contain_question_mark)
            train_dataset = train_dataset.map(normalise)
            train_dataset = train_dataset.add_column("category", ["NA"] * len(train_dataset))
            train_dataset = train_dataset.map(categorise_dataset)
            train_dataset = train_dataset.filter(remove_na_category)

            validate_dataset = validate_dataset.filter(contain_question_mark)
            validate_dataset = validate_dataset.map(normalise)
            validate_dataset = validate_dataset.add_column("category", ["NA"] * len(validate_dataset))
            validate_dataset = validate_dataset.map(categorise_dataset)
            validate_dataset = validate_dataset.filter(remove_na_category)

            comparative_dataset = load_dataset("alinet/comparativeQA", split="train")
            comparative_dataset = comparative_dataset.train_test_split(test_size=0.2)
            comparative_dataset = DatasetDict({"train": comparative_dataset["train"], "validation": comparative_dataset["test"]})

            train_dataset = concatenate_datasets(
            [train_dataset, comparative_dataset['train']]
            )

            validate_dataset = concatenate_datasets(
            [validate_dataset, comparative_dataset['validation']]
            )

            countDict = {"desc_count": 0, "recall_count": 0}

            train_dataset = train_dataset.filter(remove_excess_desc_recall_train, fn_kwargs={"countDict": countDict})

            countDict = {"desc_count": 0, "recall_count": 0}

            validate_dataset = validate_dataset.filter(remove_excess_desc_recall_validate, fn_kwargs={"countDict": countDict})

            print_distribution(train_dataset)
            print_distribution(validate_dataset)

            data = DatasetDict({"train": train_dataset, "validation": validate_dataset})



    logger.info("saving dataset")

    data["train"].to_csv(os.path.join(args.data_dir, "train.csv"))
    data["validation"].to_csv(os.path.join(args.data_dir, "validation.csv"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    datasets.logging.set_verbosity_info()
    main()
