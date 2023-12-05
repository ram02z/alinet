from pdf2image import convert_from_path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import statistics
import fitz


doc = fitz.open('./sample_data/hai_lecture_slides.pdf') # open a document
slide_text = ""
for page in doc: # iterate the document pages
  slide_text += page.get_text() # get plain text encoded as UTF-8


def compute_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    similarity_score = cosine_sim[0][1]
    return similarity_score

values = []
# iterate over transcript and compute similarity once
for i, chunk in enumerate(chunks):
    sim = compute_cosine_similarity(chunk, slide_text)
    values.append((i, sim))

# Calculate median based on sorted values
median = statistics.median([sim for i, sim in values])

removed_chunks_indexes = [i for i, sim in values if sim < median]

print("Removed chunks indexes:", removed_chunks_indexes)


