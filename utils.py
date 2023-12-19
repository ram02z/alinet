from fitz import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity_between_source(documents: list[str], source_pdf: str):
    """
    Compute similarity scores between source PDF and list of documents using cosine similarity.
    """
    with fitz.open(source_pdf, filetype="pdf") as doc:
        text = chr(12).join([page.get_text() for page in doc])

    similarity_scores = []
    for document in documents:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text, document])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        similarity_score = cosine_sim[0][1]
        similarity_scores.append(similarity_score)

    return similarity_scores
