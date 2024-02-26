import fitz 
from langchain.text_splitter import RecursiveCharacterTextSplitter
import spacy
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def pdfs_to_text(pdf_path_arr):
  res = "" 

  for pdf_path in pdf_path_arr:
    doc = fitz.open(pdf_path) 
    text = ""
    for page in doc: 
      text += page.get_text() 

    res += text
  
  return res


def compute_cosine_similarity_word_embeddings(text1, text2):
    # Load spaCy model with word embeddings
    nlp = spacy.load("en_core_web_md")

    # Get the word embeddings for each text
    #embeddings1 = nlp(text1).vector.reshape((1, -1))
    #embeddings2 = nlp(text2).vector.reshape((1, -1))

    embeddings1 = np.asarray(embeddings.embed_query(text1)).reshape((1,-1))
    embeddings2 = np.asarray(embeddings.embed_query(text2)).reshape((1,-1))
  
    # Compute cosine similarity between the two vectors
    cosine_sim = cosine_similarity(embeddings1, embeddings2)
    return cosine_sim

def return_doc_with_highest_similarity(query, documents):

  documents = [
    "I absolutely despise eating apples",
    "I hate eating honey",
    "I really love consuming apples"
  ]

  res = [None] * len(documents)
  for idx, doc in enumerate(documents):
    res[idx] = compute_cosine_similarity_word_embeddings(query, doc)
  
  print(res)

  return documents[np.argmax(res)]



if __name__ == "__main__":
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
  )

  query = "I hate eating apples"

  text = pdfs_to_text("rag/lin.pdf")
  documents = text_splitter.create_documents([text])
  
  return_doc_with_highest_similarity(query, documents)
  #lecture
  

  



