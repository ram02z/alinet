#
# ! Do not use the JSON train split from https://rajpurkar.github.io/SQuAD-explorer/
# ! It contains incorrect syntax leading to errors. Use huggingface instead. 

from datasets import load_dataset
from fastcoref import spacy_component
import spacy
import re
import json
import logging
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

squad_data = load_dataset("csv", data_files="data/balanced/train.csv")

# Testing purposes - verify whether works well on bad questions
# squad_data = load_dataset("csv", data_files="content/train_eyeball_resolved.csv")

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_md")


from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """Create a new question by reducing ambiguity in the original question by replacing common nouns and pronouns with their proper nouns in the provided context.

There are three requirements you must follow
1. There MUST be only one question
2. The question MUST NOT contain the answer
3. The new question MUST come from the provided context
4. DO NOT ANSWER THE QUESTION! """


FILE_PATH = 'questions.json'

def get_last_entries():

    if os.path.exists(FILE_PATH):
        with open(FILE_PATH, 'r') as f:
            # Parse the entire content of the file as a JSON array
            try:
                data = json.load(f)
                return data
            except json.JSONDecodeError:
                print(f"Error decoding JSON content in the file.")
    else:
        # Create an empty file with an empty array if it doesn't exist
        with open(FILE_PATH, 'a+') as initial_file:
            initial_file.seek(0)
            content = initial_file.read().strip()
            if not content:
                initial_file.write('[]')
    
    return []


def collect_question_contexts(dataset):
    print("Collecting Questions")
    print("Size of dataset:", len(dataset))
        
    question_context = [] * len(dataset)
    for index, info in enumerate(dataset):
        question_context.append((info["target"], info["source"]))

    # Number of text chunks created
    print("Number of text chunks:", len(question_context))

    return question_context


# TODO: ADD SEEKING SO THAT IN THE QUESTION_CONTEXT STORED FILE WE CONTINUE FROM WHERE WE LEFT OFF
def extract_new_questions(question_context):
    
    entries = get_last_entries()
    print("Old Entries:", len(entries))
    print("New Question Length:", len(question_context))
    start_index = 0
    new_questions = [("", False)] * len(question_context)
    for entry in entries:
        new_questions[start_index] = entry
        start_index += 1

    doc_tuples = list(nlp.pipe(question_context, as_tuples=True)) 

    print("Starting point:", start_index)
    print("Finish point:", len(doc_tuples))

    modified_questions_count = 0

    for index in range(start_index, len(doc_tuples)):

        doc, context = doc_tuples[index]
        
        contains_PROPN = False
        for token in doc:
            if token.pos_ == "PROPN":
                contains_PROPN = True
                break

        if contains_PROPN:
            new_questions[index] = ((doc.text, False)) # False specifies it was not resolved
        else:
            contains_PRON = False
            for token in doc:
                if token.pos_ == "PRON":
                    contains_PRON = True
                    break
            
            if contains_PRON:
                
                user_prompt = f"Context: {context}\n Question: {doc.text}"

                # * OpenAI API Call - Scary Stuff
                openai_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    
                ]
                )
                mod_question = openai_response.choices[0].message.content

                modified_questions_count += 1
                new_questions[index] = ((mod_question, True)) # True specifies it was resolved
            else:
                new_questions[index] = ((doc.text, False)) # False specifies it was not resolved

        json_object = json.dumps(new_questions[index], indent=4)

        with open(FILE_PATH, 'r+') as outfile:
            # Load the existing JSON array
            data = json.load(outfile)
            # Append the new question to the array
            data.append(json.loads(json_object))
            # Set the file cursor to the beginning before writing
            outfile.seek(0)
            # Write the updated JSON array back to the file
            json.dump(data, outfile, indent=4)
            

    print(len(new_questions))
    print("Number of modified questions:", modified_questions_count)

    return new_questions

def replace_questions(data, new_questions):
    question, isResolved = new_questions

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
train_data = squad_data["train"].select(range(0, 2))

# train_data = squad_data["train"]

question_context = collect_question_contexts(train_data)

new_questions = extract_new_questions(question_context)
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

# new_questions = extract_new_questions(question_context)
# if (len(train_data) != len(new_questions)):
#     print("BIG PROBLEM: Number of new questions does not match the number of original questions. Unable to map 1 to 1.")
# print("Number of new questions does match the number of original questions. Able to map 1 to 1.")

# unresolved = [""] * len(train_data)
# train_data = train_data.add_column("unresolved", unresolved)

# resolved = [""] * len(train_data)
# train_data = train_data.add_column("resolved", resolved)

# train_data = train_data.map(lambda example, index: replace_questions(example, new_questions[index]), with_indices=True)
# train_data.to_csv("./content/train.csv")