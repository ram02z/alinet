import argparse
import time
from typing import List
from datasets import Dataset, load_dataset
import datasets
import spacy
import logging
from dotenv import load_dotenv
import os
from openai import OpenAI


load_dotenv()
logger = logging.getLogger(__name__)
nlp = spacy.load("en_core_web_md")
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """Create a new question by reducing ambiguity in the original question by replacing common nouns and pronouns with their proper nouns in the provided context.

There are three requirements you must follow
1. There MUST be only one question
2. The question MUST NOT contain the answer
3. The new question MUST come from the provided context
4. DO NOT ANSWER THE QUESTION! """

def resolve_questions(examples):
    start_index = 0
    for index, example in enumerate(examples):
        if example["resolved"] == "":
            start_index = index
            break

    docs = list(
        nlp.pipe(
            [example["target"] for example in examples],
            disable=["parser", "ner", "lemmatizer"],
        )
    )

    new_dataset = []
    for index in range(0, start_index):
        example = examples[index]
        new_dataset.append(example)

    for index in range(start_index, len(docs)):
        example = examples[index]
        doc, context = docs[index], example["source"]

        contains_proper_noun = any(token.pos_ == "PROPN" for token in doc)
        contains_pronoun = any(token.pos_ == "PRON" for token in doc)
        if contains_proper_noun or not contains_pronoun:
            example.update({"resolved": example["target"]})
        else:
            user_prompt = f"Context: {context}\n Question: {doc.text}"

            # OpenAI API Call - Scary Stuff
            # openai_response = client.chat.completions.create(
            #     model="gpt-3.5-turbo",
            #     messages=[
            #         {"role": "system", "content": SYSTEM_PROMPT},
            #         {"role": "user", "content": user_prompt},
            #     ],
            # )
            # new_question = openai_response.choices[0].message.content
            new_question = "openai_response"
            example.update({"resolved": new_question})
        new_dataset.append(example)

    return Dataset.from_list(new_dataset)


def get_resolved_file_path(fp):
    dir = os.path.dirname(fp)

    filename_with_ext = os.path.basename(fp)
    filename, file_ext = os.path.splitext(filename_with_ext)

    timestamp = time.strftime("%Y-%m-%d-%H-%M")

    new_fn = f"{filename}-resolved-{timestamp}{file_ext}"

    new_fp = os.path.join(dir, new_fn)

    return new_fp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file_path", help="Path to CSV file to run coreference resolution on"
    )

    args = parser.parse_args()
    data = load_dataset("csv", data_files=args.file_path, split="train")
    column_names: List[str] = data.column_names
    if "resolved" not in column_names:
        logger.info("'resolved' column not in dataset")
        data = data.add_column(name="resolved", column=len(data) * [""])

    # TODO: add exception handling
    data = resolve_questions(data)
    data = data.remove_columns("target")
    data = data.rename_column("resolved", "target")

    # TODO: refactor
    # Assume this means all is resolved so the target column
    # data = data.map(
    #     lambda example, index: replace_questions(example, new_questions[index]),
    #     with_indices=True,
    # )
    data.to_csv(get_resolved_file_path(args.file_path))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    datasets.logging.set_verbosity_info()
    main()
