import argparse
import time
from typing import List
from datasets import Dataset, load_dataset
import datasets
import spacy
import logging
from dotenv import load_dotenv
import os
import openai
from openai import OpenAI


load_dotenv()
logger = logging.getLogger(__name__)
nlp = spacy.load("en_core_web_md")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """Given the context provided, generate a new question by reducing ambiguity in the original question.

The structure of the input is as follows:

{context} <Qsep> {original question}

Ensure the following:
1. Formulate only one question.
2. The question should not include the answer.
3. The new question must arise from the given context.
4. Maintain the same initial word as the original question.
5. DO NOT PROVIDE THE ANSWER!"""


def resolve_questions(examples, fp):

    modified_questions_count = 0
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

    print("Start Index:", start_index)
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
            user_prompt = f"{context} <Qsep> {doc.text}"
            modified_questions_count += 1
            # If there is an error we return the program and save the current dataset
            try:
                # OpenAI API Call - Scary Stuff
                openai_response = client.chat.completions.create(
                    model="gpt-4-1106-preview",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    seed=1,
                    temperature=0,
                )
                new_question = openai_response.choices[0].message.content

                # new_question = "OPEN_AI_RESPONSE"
                example.update({"resolved": new_question})

            except openai.APIError as e:
                logger.fatal(f"Open AI Error: {e.message}\nFailed at question index: {index}")
                
                # Leftover Examples
                new_dataset.append(example)
                for index in range(index+1, len(examples)):
                    example = examples[index]
                    new_dataset.append(example)

                data = Dataset.from_list(new_dataset)
                data.to_csv(get_resolved_file_path(fp))
                exit(1)
        
        new_dataset.append(example)
        
    print("Number of modified question:", modified_questions_count)

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

    data = resolve_questions(data, args.file_path)
    # data = data.remove_columns("target")
    # data = data.rename_column("resolved", "target")

    data.to_csv(get_resolved_file_path(args.file_path))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    datasets.logging.set_verbosity_info()
    main()
