import pickle
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from video_processing import slide_chunking

def compute_cosine_similarity_word_embeddings(text1, text2):
    # Load spaCy model with word embeddings
    nlp = spacy.load("en_core_web_md")

    # Get the word embeddings for each text
    embeddings1 = nlp(text1).vector.reshape(1, -1)
    embeddings2 = nlp(text2).vector.reshape(1, -1)

    # Compute cosine similarity between the two vectors
    cosine_sim = cosine_similarity(embeddings1, embeddings2)[0, 0]
    return cosine_sim

def get_similarity_scores(duration, transcript_chunks, video_path, slide_path):

    slide_chunks = slide_chunking(video_path, slide_path)

    i = 0
    similarity_scores = []
    for j, chunk in enumerate(transcript_chunks):
        # ensure we only process transcript chunks that occour before the final slide
        if chunk['timestamp'][0] < duration:
            list_of_slide_indices = []
            while i < len(slide_chunks):
                list_of_slide_indices.append(i)
                if chunk['timestamp'][1] <= slide_chunks[i][2]:
                    transcript_chunk_text = chunk['text']

                    # aggregate all the text from different slides, if more than one, into a single var
                    slide_text = ""
                    for index in list_of_slide_indices:
                        slide_text += slide_chunks[index][0]

                    # compute sim between the text retrieved from slide and the corresponding slide's text
                    cosine_sim = compute_cosine_similarity_word_embeddings(transcript_chunk_text, slide_text)

                    similarity_scores.append(cosine_sim)
                    
                    break
                i += 1
    return similarity_scores
    