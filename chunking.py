from transformers import AutoTokenizer
from qg import Model
import spacy

INPUT_TOKEN_LIMIT = {Model.DISCORD: 1024}

nlp = spacy.load("en_core_web_md")
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
        stride_length=80,
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

        # Remove sentences with < 4 words
        for chunk in time_chunks:
            sentenceArr = chunk["text"]
            chunk["text"] = [
                sentence
                for sentence in sentenceArr
                if len(sentence.split()) >= 4 and "..." not in sentence
            ]

        # Merge consecutive sentences within a chunk when the second sentence starts with a coordinating conjunction ('CCONJ').
        for chunk in time_chunks:
            chunk_sentences = chunk["text"]
            for i in range(len(chunk_sentences) - 1, 0, -1):
                current_sent = chunk_sentences[i]
                prev_sent = chunk_sentences[i - 1]
                doc = nlp(current_sent)
                if doc[0].pos_ == "CCONJ":
                    merged_sentence = prev_sent + " " + current_sent
                    chunk_sentences[i - 1] = merged_sentence
                    chunk_sentences.pop(i)


        # Add stride to chunks
        # NOTE: In the future, look into adding right stride to potentially only the first issue.
        # Cannot add right stride to all chunks because else we cause some chunks to start with a coordinating conjunctive
        chunks_with_stride = []
        for chunk_idx in range(len(time_chunks)):
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
                    "text": " ".join(left_sents + time_chunks[chunk_idx]["text"]),
                }
            )

        return chunks_with_stride
