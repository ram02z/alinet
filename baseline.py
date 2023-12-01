import pickle
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BartTokenizer, T5ForConditionalGeneration, T5Tokenizer

ending_punct = ['.', '!', '?']


def gen_t5_squad2(chunks):
    chunks_questions = {}

    model_name = "allenai/t5-small-squad2-question-generation"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    for chunk in chunks:
        input_ids = tokenizer.encode(chunk['text'], return_tensors="pt")
        res = model.generate(input_ids)
        generated_questions = tokenizer.batch_decode(res, skip_special_tokens=True)
        chunks_questions[chunk['text']] = generated_questions[0]

    return chunks_questions


def gen_bart_discord(chunks):
    chunks_questions = {}

    qg_tokenizer = AutoTokenizer.from_pretrained("Salesforce/discord_qg")
    qg_model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/discord_qg")

    for chunk in chunks:
        encoder_ids = qg_tokenizer.batch_encode_plus([chunk['text']], add_special_tokens=True, padding=True,
                                                     truncation=True, return_tensors="pt")
        model_output = qg_model.generate(**encoder_ids)
        generated_questions = qg_tokenizer.batch_decode(model_output, skip_special_tokens=True)

        chunks_questions[chunk['text']] = generated_questions[0]

    return chunks_questions


def gen_bart_nq(chunks):
    chunks_questions = {}

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    model = AutoModelForSeq2SeqLM.from_pretrained("McGill-NLP/bart-qg-nq-checkpoint")

    for chunk in chunks:
        inputs = tokenizer([chunk['text']], return_tensors='pt')
        question_ids = model.generate(inputs['input_ids'], num_beams=5, early_stopping=True)
        generated_questions = tokenizer.batch_decode(question_ids, skip_special_tokens=True)

        chunks_questions[chunk['text']] = generated_questions[0]

    return chunks_questions


def gen_bart_eqg(chunks):
    chunks_questions = {}

    qg_tokenizer = AutoTokenizer.from_pretrained("voidful/bart-eqg-question-generator")
    qg_model = AutoModelForSeq2SeqLM.from_pretrained("voidful/bart-eqg-question-generator")

    for chunk in chunks:
        inputs = qg_tokenizer([chunk['text']], return_tensors='pt')

        question_ids = qg_model.generate(inputs['input_ids'], num_beams=5, early_stopping=True)
        generated_questions = qg_tokenizer.batch_decode(question_ids, skip_special_tokens=True)

        chunks_questions[chunk['text']] = generated_questions[0]

    return chunks_questions

def gen_bart_unknown(chunks):
    chunks_questions = {}

    qg_tokenizer = AutoTokenizer.from_pretrained("voidful/context-only-question-generator")
    qg_model = AutoModelForSeq2SeqLM.from_pretrained("voidful/context-only-question-generator")

    for chunk in chunks:
        inputs = qg_tokenizer([chunk['text']], return_tensors='pt')

        question_ids = qg_model.generate(inputs['input_ids'], num_beams=5, early_stopping=True)
        generated_questions = qg_tokenizer.batch_decode(question_ids, skip_special_tokens=True)

        chunks_questions[chunk['text']] = generated_questions[0]

    return chunks_questions

def gen_gpt3_5(chunks):
    questions_list = [
        'What is the main topic of the lecture?',
        "What is the speaker's intention?",
        'Can you estimate the level of detail the lecture will cover regarding natural language processing?',
        'What is the speaker asking the audience to do?',
        'What is the primary focus of the discussion?',
        'What does the speaker inquire about regarding the estimate?',
        'What is the connection between smartphones and embedded AIs?',
        'What are the various subsystems mentioned that utilize AI in smartphones?',
        'Why is language considered a key element in interaction?',
        'What distinction is made between natural languages and formal languages like Python or C++?',
        'What fields contribute to natural language processing at its core?',
        'What are the two main subfields mentioned within NLP?',
        'What role does natural language understanding play in NLP?',
        'What does the speaker present as the motivation for natural language understanding?',
        'What types of insights can be gained from analyzing large amounts of text data?',
        'What are some common applications of natural language generation systems?',
        'What does the speaker present as the evolution of NLP systems over time?',
        'What are the main types of early NLP systems mentioned?',
        'What role did classical machine learning algorithms play in the development of NLP?',
        'What are the advancements mentioned that characterize the current state of NLP research?',
        'What are some common subfields of NLP?',
        'What is the significance of considering documents and corpora in the context of NLP?',
        'What are the primary steps in the NLP pre-processing pipeline?',
        'How does annotation contribute to the enrichment of tokens?',
        'What is the distinction between lemmatization and stemming in terms of processing words?',
        'Why is filtering important in the NLP pre-processing pipeline?',
        'What are n-grams, and why might they be valuable in NLP?',
        'What paper is recommended for reading?',
        'How does spelling correction in Microsoft Word relate to NLP concepts?',
        'What is the difference between lemmatization and stemming?',
        'How does the time-saving aspect influence the choice between lemmatization and stemming?',
        'How does filtering contribute to the efficiency of NLP models?',
        'How is the process of data analysis influenced by the need to save time in NLP?',
        'What is the role of dictionaries in lemmatization?',
        'Where can students find information about upcoming lab sessions and coursework on Moodle?'
    ]

    chunks_questions = {}

    for i, chunk in enumerate(chunks):
        chunks_questions[chunk['text']] = questions_list[i]

    return chunks_questions


with open('./test_data/comp3074_lecture_2.pkl', 'rb') as file:
    chunks = pickle.load(file)
    time_chunks = []
    current_sentence = ""
    start_timestamp = None
    min_duration = 60

    if not chunks['chunks'][-1]['timestamp'][1]:
        chunks['chunks'][-1]['timestamp'] = (chunks['chunks'][-1]['timestamp'][0], chunks['chunks'][-1]['timestamp'][0])

    for chunk in chunks['chunks']:
        text = chunk['text']
        timestamp = chunk['timestamp']

        if start_timestamp is None:
            # Start a new sentence
            start_timestamp = timestamp[0]

        current_sentence += text

        sentence_completed = text.strip()[-1] in ending_punct
        time_elapsed = (timestamp[1] - start_timestamp) >= min_duration

        # TODO: tokenize to ensure below token limit for qg model
        if sentence_completed and time_elapsed:
            time_chunks.append({'timestamp': (start_timestamp, timestamp[1]), 'text': current_sentence.strip()})
            current_sentence = ""
            start_timestamp = None

    if current_sentence.strip():
        time_chunks.append(
            {'timestamp': (start_timestamp, chunks['chunks'][-1]['timestamp'][1]), 'text': current_sentence.strip()})

models_questions = {
    'bart_unknown': gen_bart_unknown(time_chunks),
    't5_squad2': gen_t5_squad2(time_chunks),
    'bart_discord': gen_bart_discord(time_chunks),
    'bart_nq': gen_bart_nq(time_chunks),
    'bart_eqg': gen_bart_eqg(time_chunks),
    'gpt3': gen_gpt3_5(time_chunks),
}

chunks_by_model = {}

for model, chunk_questions in models_questions.values():
    for chunk, question in chunk_questions.values():
        chunks_by_model[chunk].update({model: question})

print(chunks_by_model)