from transformers import PreTrainedTokenizer
import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import statistics


class BaseChunker:
    def __init__(
        self, chunks: list[dict[str | tuple[float, float]]], audio_length: float
    ):
        """
        :param chunks: transcript chunks with chunk-level timestamps
        :param audio_length: length of the original audio in seconds
        """

        # Last chunk end timestamp could be missing
        if not chunks[-1]["timestamp"][1]:
            chunks[-1]["timestamp"] = (
                chunks[-1]["timestamp"][0],
                audio_length,
            )

        self.chunks = chunks
        self._audio_length = audio_length

    def filter(self, pdf_stream: bytes):
        """
        :param pdf_stream: supplementary pdf to use for chunk filtering
        :return: transcript chunks with chunk-level timestamps
        """
        with fitz.open(stream=pdf_stream, filetype="pdf") as doc:
            text = chr(12).join([page.get_text() for page in doc])

        similarity_scores = []
        for chunk in self.chunks:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([text, chunk["text"]])
            cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
            similarity_score = cosine_sim[0][1]
            similarity_scores.append(similarity_score)

        median = statistics.median(similarity_scores)

        self.chunks = [
            chunk for sim, chunk in zip(similarity_scores, self.chunks) if sim > median
        ]

    def chunk(self, sentence_overlap: int):
        """
        :param sentence_overlap: number of sentence to prepend and append to each chunk
        """
        raise NotImplementedError


class TokenChunker(BaseChunker):
    def __init__(self, tokenizer: PreTrainedTokenizer, token_limit: int, **kwargs):
        """
        :param tokenizer: tokenizer used for downstream QG model
        :param token_limit: the maximum input token limit for the downstream QG model
        """
        super().__init__(**kwargs)
        self._tokenizer = tokenizer
        self._token_limit = token_limit

    def chunk(self, sentence_overlap=0):
        time_chunks = []
        current_sentence = ""
        start_timestamp = None
        max_tokens = self._token_limit

        for chunk in self.chunks:
            text = chunk["text"]
            timestamp = chunk["timestamp"]

            if start_timestamp is None:
                start_timestamp = timestamp[0]

            potential_sentence = current_sentence + text
            token_count = len(self._tokenizer.tokenize(potential_sentence))

            if token_count > max_tokens and current_sentence.strip()[-1] == ".":
                time_chunks.append(
                    {
                        "timestamp": (start_timestamp, timestamp[1]),
                        "text": current_sentence.strip(),
                    }
                )
                current_sentence = ""
                start_timestamp = None
            else:
                current_sentence = potential_sentence

        if current_sentence.strip():
            time_chunks.append(
                {
                    "timestamp": (
                        start_timestamp,
                        self.chunks[-1]["timestamp"][1],
                    ),
                    "text": current_sentence.strip(),
                }
            )

        self.chunks = time_chunks


class TimeChunker(BaseChunker):
    def __init__(self, min_duration: int = 60, **kwargs):
        """
        :param min_duration: minimum duration for each chunk
        """
        super().__init__(**kwargs)
        self._min_duration = min_duration

    def chunk(self, sentence_overlap=0):
        time_chunks = []
        current_sentence = ""
        start_timestamp = None

        for chunk in self.chunks:
            text = chunk["text"]
            timestamp = chunk["timestamp"]

            if start_timestamp is None:
                start_timestamp = timestamp[0]

            current_sentence += text

            sentence_completed = text.strip()[-1] == "."
            time_elapsed = (timestamp[1] - start_timestamp) >= self._min_duration

            if sentence_completed and time_elapsed:
                time_chunks.append(
                    {
                        "timestamp": (start_timestamp, timestamp[1]),
                        "text": current_sentence.strip(),
                    }
                )
                current_sentence = ""
                start_timestamp = None

        if current_sentence.strip():
            time_chunks.append(
                {
                    "timestamp": (start_timestamp, self.chunks[-1]["timestamp"][1]),
                    "text": current_sentence.strip(),
                }
            )

        self.chunks = time_chunks


if __name__ == "__main__":
    import pickle
    from transformers import BartTokenizer

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    with open("experiments/qg/comp3074_lecture_2.pkl", "rb") as file:
        chunks = pickle.load(file)["chunks"]
        chunker = TimeChunker(min_duration=60, chunks=chunks, audio_length=2319)
        chunker.chunk(chunks)
        with open("slides.pdf", "rb") as pdf_file:
            chunker.filter(pdf_file.read())
