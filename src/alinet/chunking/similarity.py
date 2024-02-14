import spacy
from sklearn.metrics.pairwise import cosine_similarity
import warnings


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


def get_similarity_scores(duration, transcript_chunks, slide_chunks):
    """
    Get similarity scores between transcript chunks and corresponding slide chunks.

    Parameters:
    - duration (float): The duration until which transcript chunks are considered.
    - transcript_chunks (list): List of transcript chunks.
    - slide_chunks (list): List of slide chunks.

    Returns:
    list: A list of similarity scores between transcript chunks and corresponding slide chunks.
    """
    i = 0
    similarity_scores = []
    for chunk in transcript_chunks:
        # ensure we only process transcript chunks that occur before the final slide
        if chunk["timestamp"][0] < duration:
            list_of_slide_indices = []
            while i < len(slide_chunks):
                list_of_slide_indices.append(i)
                if chunk["timestamp"][1] <= slide_chunks[i][2]:
                    transcript_chunk_text = chunk["text"]

                    # aggregate all the text from different slides, if more than one, into a single var
                    slide_text = ""
                    for index in list_of_slide_indices:
                        slide_text += slide_chunks[index][0]

                    # compute sim between the text retrieved from slide and the corresponding slide's text
                    cosine_sim = compute_cosine_similarity_word_embeddings(
                        transcript_chunk_text, slide_text
                    )

                    similarity_scores.append(cosine_sim)

                    break
                i += 1
    return similarity_scores


def filter_questions_by_retention_rate(
    sim_scores, generated_questions, similarity_threshold, filtering_threshold
):
    """
    Filter questions based on the retention rate and similarity threshold.

    Parameters:
    - sim_scores (list): List of similarity scores.
    - generated_questions (list): List of generated questions.
    - similarity_threshold (float): Threshold for similarity scores.
    - filtering_threshold (float): Threshold for the retention rate.

    Returns:
    list: A list of filtered questions based on the retention rate and similarity threshold.
    """
    scores_and_questions = zip(sim_scores, generated_questions)
    filtered_questions = [
        question for sim, question in scores_and_questions if sim > similarity_threshold
    ]
    retention_rate = len(filtered_questions) / len(generated_questions)

    if retention_rate < filtering_threshold:
        warnings.warn(
            "Could not effectively perform question filtering, all generated questions are being returned"
        )
        return generated_questions
    else:
        return filtered_questions
