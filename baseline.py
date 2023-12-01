import pickle
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

qg_tokenizer = AutoTokenizer.from_pretrained("Salesforce/discord_qg")
qg_model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/discord_qg")

ending_punct = ['.', '!', '?']

with open('./sample_data/lecture_1/audio.pkl', 'rb') as file:
    chunks = pickle.load(file)
    time_chunks = []
    current_sentence = ""
    start_timestamp = None
    min_duration = 60

    for chunk in chunks:
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
            {'timestamp': (start_timestamp, chunks[-1]['timestamp'][1]), 'text': current_sentence.strip()})

    for chunk in time_chunks:
        for start_word in ["How", "Why", "When", "What"]:
            encoder_ids = qg_tokenizer.batch_encode_plus([chunk['text']], add_special_tokens=True, padding=True,
                                                         truncation=True, return_tensors="pt")
            decoder_input_ids = \
                qg_tokenizer.batch_encode_plus([start_word], add_special_tokens=True, return_tensors="pt")["input_ids"][
                :,
                :-1]
            model_output = qg_model.generate(**encoder_ids, decoder_input_ids=decoder_input_ids, max_length=20)
            generated_questions = qg_tokenizer.batch_decode(model_output, skip_special_tokens=True)

            print(generated_questions)
