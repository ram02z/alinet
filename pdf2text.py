from pdf2image import convert_from_path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pytesseract
import statistics
import spacy

# PDF to convert to text
pdf_images = convert_from_path('./sample_data/hai_lecture_slides.pdf')

# slide to text conversion
slide_text = ""
for img in pdf_images:
    slide_text += (pytesseract.image_to_string(img,lang='eng'))


def compute_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    similarity_score = cosine_sim[0][1]
    return similarity_score


values = []
# Compute similarity between chunks and slide text
for i, chunk in enumerate(chunks):
    sim = compute_cosine_similarity(chunk, slide_text)
    values.append((i, sim))

median = statistics.median(values, key=lambda x: x[1])

# Retrieve indexes of chunks to be removed
removed_chunks_indexes = [i for i, sim in values if sim < median]
print(removed_chunks_indexes)

