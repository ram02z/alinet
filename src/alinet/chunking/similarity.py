import spacy
from sklearn.metrics.pairwise import cosine_similarity
import warnings

from alinet.chunking.model import TimeChunk


def compute_cosine_similarity_word_embeddings(text1, text2):
    """
    Compute the cosine similarity between the word embeddings of two texts.

    Parameters:
    - text1 (str): The first text.
    - text2 (str): The second text.

    Returns:
    float: The cosine similarity between the word embeddings of the two texts.
    """
    # Load spaCy model with word embeddings
    nlp = spacy.load("en_core_web_md")

    # Get the word embeddings for each text
    embeddings1 = nlp(text1).vector.reshape((1, -1))
    embeddings2 = nlp(text2).vector.reshape((1, -1))

    # Compute cosine similarity between the two vectors
    cosine_sim = cosine_similarity(embeddings1, embeddings2)[0, 0]
    return cosine_sim


def find_matching_slide_range(
    chunk: TimeChunk, slide_chunks: list[TimeChunk]
) -> tuple[int, int]:
    """
    Finds the range of slide indices corresponding to the given transcript chunk.
    """

    start_index = 0
    end_index = 0
    for i, slide in enumerate(slide_chunks):
        if chunk.start_time >= slide.start_time and chunk.end_time <= slide.end_time:
            start_index = i
            end_index = i
            break
        elif chunk.start_time < slide.end_time:
            start_index = i
        elif chunk.end_time > slide.end_time:
            end_index = i
    return start_index, end_index + 1  # Adjusted to get the correct range


def get_similarity_scores(
    transcript_chunks: list[TimeChunk], slide_chunks: list[TimeChunk]
) -> list[float]:
    """
    Get similarity scores between transcript chunks and corresponding slide chunks.

    Parameters:
    - transcript_chunks (list): List of transcript chunks.
    - slide_chunks (list): List of slide chunks.

    Returns:
    list: A list of similarity scores between transcript chunks and corresponding slide chunks.
    """
    duration = transcript_chunks[-1].end_time
    similarity_scores = []

    for chunk in transcript_chunks:
        if chunk.start_time < duration:
            start_index, end_index = find_matching_slide_range(chunk, slide_chunks)
            slide_text = " ".join(
                slide_chunks[i].text for i in range(start_index, end_index)
            )
            similarity_scores.append(
                compute_cosine_similarity_word_embeddings(chunk.text, slide_text)
            )

    return similarity_scores


def filter_questions_by_retention_rate(
    sim_scores: list[float],
    generated_questions: list[str],
    similarity_threshold: float,
    filtering_threshold: float,
) -> dict[int, str]:
    """
    Filter questions based on the retention rate and similarity threshold.

    Parameters:
    - sim_scores (list): List of similarity scores.
    - generated_questions (list): List of generated questions.
    - similarity_threshold (float): Threshold for similarity scores.
    - filtering_threshold (float): Threshold for the retention rate.

    Returns:
    dict: A Dictionary of filtered questions based on the retention rate and similarity threshold.
    """
    filtered_questions = {}

    for index, (sim, question) in enumerate(zip(sim_scores, generated_questions)):
        if sim > similarity_threshold:
            filtered_questions[index] = question

    retention_rate = len(filtered_questions) / len(generated_questions)

    if retention_rate < filtering_threshold:
        warnings.warn(
            "Could not effectively perform question filtering, all generated questions are being returned"
        )
        return {idx: question for idx, question in enumerate(generated_questions)}
    else:
        return filtered_questions
