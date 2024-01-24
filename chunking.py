from transformers import AutoTokenizer
from qg import Model
from spacy.lang.en import English

INPUT_TOKEN_LIMIT = {Model.DISCORD: 1024}

nlp = English()
nlp.add_pipe("sentencizer")


class ChunkPipeline:
    def __init__(
        self,
        model_id=Model.DISCORD,
    ):
        """
        :param model_id: name of huggingface transformers model
        """

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._input_limit = INPUT_TOKEN_LIMIT.get(model_id, 512)

    def __call__(
        self,
        chunks: list[dict[str, str | tuple[float, float]]],
        audio_length: float,
        stride_length=50,
        min_duration=120,
    ) -> list[dict[str, str | tuple[float, float]]]:
        """
        :param chunks: transcript chunks with chunk-level timestamps
        :param audio_length: length of the original audio in seconds
        :param stride_length: maximum number of tokens to add to both sides of chunk
        :param min_duration: minimum duration per chunk
        """

        # Last chunk end timestamp could be missing
        if not chunks[-1]["timestamp"][1]:
            chunks[-1]["timestamp"] = (
                chunks[-1]["timestamp"][0],
                audio_length,
            )

        time_chunks = []
        current_sentence = ""
        start_timestamp = None
        chunk_limit = self._input_limit - 2 * stride_length

        def process_chunk(chunk):
            nonlocal current_sentence, start_timestamp
            doc = nlp(current_sentence)
            time_chunks.append(
                {
                    "timestamp": (start_timestamp, chunk["timestamp"][1]),
                    "text": [str(sent).strip() for sent in doc.sents],
                }
            )
            current_sentence = ""
            start_timestamp = None

        for chunk in chunks:
            text, timestamp = chunk["text"], chunk["timestamp"]

            if start_timestamp is None:
                start_timestamp = timestamp[0]

            current_sentence += text
            token_count = len(self._tokenizer.tokenize(current_sentence))

            if current_sentence.strip()[-1] == "." and (
                token_count > chunk_limit
                or (timestamp[1] - start_timestamp) >= min_duration
            ):
                process_chunk(chunk)

        # Add left over sentence(s) to last chunk
        if current_sentence.strip():
            process_chunk(chunks[-1])

        # remove sentences with < 4 words
        for chunk in time_chunks:
            sentenceArr = chunk["text"]
            chunk["text"] = [
                sentence
                for sentence in sentenceArr
                if len(sentence.split()) >= 4 and "..." not in sentence
            ]

        # Add stride to chunks
        chunks_with_stride = []
        for chunk_idx in range(len(time_chunks)):
            # Right stride
            right_sents = []
            if chunk_idx < len(time_chunks) - 1:
                next_chunk_idx = chunk_idx + 1
                sentence_idx = 0
                while len(time_chunks[next_chunk_idx]["text"]) > sentence_idx:
                    sent = time_chunks[next_chunk_idx]["text"][sentence_idx]
                    token_count = len(
                        self._tokenizer.tokenize("".join(right_sents) + sent)
                    )
                    if token_count >= stride_length:
                        break
                    right_sents.append(sent)
                    sentence_idx += 1

            # Left stride
            left_sents = []
            if chunk_idx > 0:
                prev_chunk_idx = chunk_idx - 1
                sentence_index = 1
                while len(time_chunks[prev_chunk_idx]["text"]) > sentence_index:
                    sent = time_chunks[prev_chunk_idx]["text"][-sentence_index]
                    token_count = len(
                        self._tokenizer.tokenize("".join(left_sents) + sent)
                    )
                    if token_count >= stride_length:
                        break
                    left_sents.append(sent)
                    sentence_index += 1

            chunks_with_stride.append(
                {
                    "timestamp": time_chunks[chunk_idx]["timestamp"],
                    "text": " ".join(
                        left_sents + time_chunks[chunk_idx]["text"] + right_sents
                    ),
                }
            )

        return chunks_with_stride
