#
# ! Do not use the JSON train split from https://rajpurkar.github.io/SQuAD-explorer/
# ! It contains incorrect syntax leading to errors. Use huggingface instead. 

from datasets import load_dataset
from fastcoref import spacy_component
import spacy
import re
import json
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def contain_question_mark(data):
    return data["target"][-1].rstrip() == "?"

def normalise(data):
    # Lowercase the text
    # data["source"] = data["source"].lower()
    # data["target"] = data["target"].lower()

    # Remove new line characters
    data["source"] = data["source"].replace("\n", " ")

    return data

squad_data = (
        load_dataset("squad")
        .select_columns(["context", "question"])
        .rename_columns({"context": "source", "question": "target"})
        .filter(contain_question_mark)
        .map(normalise)
    )

nlp = spacy.load("en_core_web_md")
nlp.add_pipe(
    "fastcoref",
    config={
        'model_architecture': 'LingMessCoref',
        'model_path': 'biu-nlp/lingmess-coref',
        'device': 'cuda:0'
        }
    )

def collect_question_contexts(dataset):
    print("Collecting Questions")
        
    # Number of questions in dataset
    print(len(dataset))
    uniqueContext = dataset[0]["source"]
    questions = ""
    texts = []
    for info in dataset:
        context = info["source"]
        question = info["target"]

        # We want to have a one to many mapping (one context to many questions) to improve performance in spaCy/LingMessCoref
        # If its a new context then we have collected all the questions for the previous context
        if context != uniqueContext:
            text = f"""{uniqueContext} : {questions}"""
            texts.append(text)
            uniqueContext = context
            questions = ""
        questions += f"""<sqSEP> {question} </sqSEP> """

    # Fixes issue with the last context and questions in the dataset not being added
    text = f"""{uniqueContext} : {questions}"""
    texts.append(text)

    # Number of text chunks created
    print(len(texts))

    return texts


def extract_new_questions(texts):
    print("Extract New Questions")

    # Batch Inference -> Can't improve in terms of speed
    # This gives us a list of documents with the 
    docs = list(nlp.pipe(texts, component_cfg={"fastcoref": {'resolve_text': True}})) 

    # Extract all content between the <sqSEP> and </sqSEP> tags without leading/trailing whitespaces
    new_questions = []
    start_tag = "<sqSEP>"
    end_tag = "</sqSEP>"

    numberOfQuestions = 0
    for doc in docs:

        # Just accept wtf is happening in this section
        matches = re.finditer(f'{re.escape(start_tag)}\s*(.*?)\s*{re.escape(end_tag)}', doc.text, re.DOTALL)
        og_questions = [match.group(1) for match in matches]

        matches = re.finditer(f'{re.escape(start_tag)}\s*(.*?)\s*{re.escape(end_tag)}', doc._.resolved_text, re.DOTALL)
        mod_questions = [match.group(1) for match in matches]

        question_docs = list(nlp.pipe(og_questions))
        index = 0

        for q_doc in question_docs:
            contains_PROPN = False
            for token in q_doc:
                if token.pos_ == "PROPN":
                    contains_PROPN = True
                    break

            if contains_PROPN:
                new_questions.append((q_doc.text, False)) # False specifies it was not resolved
            else:
                contains_PRON = False
                for token in q_doc:
                    if token.pos_ == "PRON":
                        contains_PRON = True
                        break
                
                if contains_PRON:
                    new_questions.append((mod_questions[index], True)) # True specifies it was resolved
                else:
                    new_questions.append((q_doc.text, False)) # False specifies it was not resolved
            
            index += 1
        

    print(len(new_questions))

    return new_questions

def replace_questions(data, new_questions):

    question = new_questions[0]
    isResolved = new_questions[1]

    if isResolved:
        # If question was modified but its the same as the original target put it in unresolved
        if question == data['target']:
            data['unresolved'] = data['target']
        else:
            data['resolved'] = question
    else:
        data['unresolved'] = question

    return data

print("Starting Train...")

# Testing purposes
# train_data = squad_data["train"].select(range(19020, 20020))
# train_data = squad_data["train"].select(range(0, 100))
# train_data = squad_data["train"].select(range(len(squad_data["train"]) - 10, len(squad_data["train"])))
# validation_data = squad_data["validation"].select(range(len(squad_data["validation"]) - 10, len(squad_data["validation"])))

train_data = squad_data["train"]

question_contexts = collect_question_contexts(train_data)
new_questions = extract_new_questions(question_contexts)
if (len(train_data) != len(new_questions)):
    print("BIG PROBLEM: Number of new questions does not match the number of original questions. Unable to map 1 to 1.")
print("Number of new questions does match the number of original questions. Able to map 1 to 1.")

unresolved = [""] * len(train_data)
train_data = train_data.add_column("unresolved", unresolved)

resolved = [""] * len(train_data)
train_data = train_data.add_column("resolved", resolved)

train_data = train_data.map(lambda example, index: replace_questions(example, new_questions[index]), with_indices=True)
train_data.to_csv("./content/train.csv")


# validation_data = squad_data["validation"]
# question_contexts = collect_question_contexts(validation_data)
# new_questions = extract_new_questions(question_contexts)
# if (len(train_data) != len(new_questions)):
#     print("BIG PROBLEM: Number of new questions does not match the number of original questions. Unable to map 1 to 1.")
# print("Number of new questions does match the number of original questions. Able to map 1 to 1.")

# unresolved = [""] * len(train_data)
# train_data = train_data.add_column("unresolved", unresolved)

# resolved = [""] * len(train_data)
# train_data = train_data.add_column("resolved", resolved)

# train_data = train_data.map(lambda example, index: replace_questions(example, new_questions[index]), with_indices=True)
# train_data.to_csv("./content/train.csv")
