from transformers import AutoTokenizer
import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import statistics
from qg import Model

INPUT_TOKEN_LIMIT = {Model.DISCORD: 1024}


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
        pdf_stream: bytes = None,
        sentence_overlap=0,
        min_duration=60,
    ) -> list[dict[str, str | tuple[float, float]]]:
        """
        :param chunks: transcript chunks with chunk-level timestamps.
        :param audio_length: length of the original audio in seconds
        :param pdf_stream: supplementary pdf to use for chunk filtering
        :param sentence_overlap: number of sentence to prepend and append to each chunk
        """

        # Last chunk end timestamp could be missing
        if not chunks[-1]["timestamp"][1]:
            chunks[-1]["timestamp"] = (
                chunks[-1]["timestamp"][0],
                audio_length,
            )

        self._chunks = chunks
        self._chunk(sentence_overlap, min_duration)
        if pdf_stream:
            self._filter(pdf_stream)
        return self._chunks

    def _filter(self, pdf_stream):
        with fitz.open(stream=pdf_stream, filetype="pdf") as doc:
            text = chr(12).join([page.get_text() for page in doc])

        similarity_scores = []
        for chunk in self._chunks:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([text, chunk["text"]])
            cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
            similarity_score = cosine_sim[0][1]
            similarity_scores.append(similarity_score)

        median = statistics.median(similarity_scores)

        self._chunks = [
            chunk for sim, chunk in zip(similarity_scores, self._chunks) if sim > median
        ]

    # TODO: implement sentence overlap
    def _chunk(self, sentence_overlap, min_duration):
        time_chunks = []
        current_sentence = ""
        start_timestamp = None

        for chunk in self._chunks:
            text = chunk["text"]
            timestamp = chunk["timestamp"]

            if start_timestamp is None:
                start_timestamp = timestamp[0]

            current_sentence += text
            token_count = len(self._tokenizer.tokenize(current_sentence))

            if current_sentence.strip()[-1] == "." and (
                token_count > self._input_limit
                or (timestamp[1] - start_timestamp) >= min_duration
            ):
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
                    "timestamp": (
                        start_timestamp,
                        self._chunks[-1]["timestamp"][1],
                    ),
                    "text": current_sentence.strip(),
                }
            )

        self._chunks = time_chunks
