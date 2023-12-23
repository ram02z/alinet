import pickle
import qg
from chunking import ChunkPipeline
from fitz import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0, 1]

# get the whisper chunks
with open("experiments/qg/comp3074_lecture_2.pkl", "rb") as file:
    whisper_chunks = pickle.load(file)['chunks']

# get the slide chunks with timestamps
with open('slide_chunks.pkl', 'rb') as file:
    slide_chunks = pickle.load(file)

# generate trasncript chunks
qg_model = qg.Model.DISCORD
chunk_pipe = ChunkPipeline(qg_model)
transcript_chunks = chunk_pipe(whisper_chunks,2301)

# get last endtime in the slide chunkss
endtime = slide_chunks[-1][2]

i = 0

# a list of the indices of the relevant chunks
relevant_chunks = []
for j, chunk in enumerate(transcript_chunks):
    # ensure we only process transcript chunks that correspond to the slide chunks
    if chunk['timestamp'][0] < endtime:
        list_of_slide_indices = []
        while i < len(slide_chunks):
            list_of_slide_indices.append(i)
            if chunk['timestamp'][1] <= slide_chunks[i][2]:
                # print("FITS")
                # print(f"current chunks start time is: {chunk['timestamp'][0]} and the end time is: {chunk['timestamp'][1]}", i)
                # print(f"The slide chunk currently as an end time of: {slide_chunks[i][2]}")
                # print(f"the slide indices that need to be combined are: {list_of_slide_indices}")
                # print("===============================================================")
                
                transcript_chunk_text = chunk['text']
                print(f"The text on the chunk is: {transcript_chunk_text}")
                print("-------------------------------------------------------------------------------")

                slide_text = ""
                for index in list_of_slide_indices:
                    slide_text += slide_chunks[index][0]

                print(f"The text on the relevant slides is: {slide_text}")
                # compute similarity between the transcript text and the relevant slides
                cosine_sim = compute_cosine_similarity(transcript_chunk_text, slide_text)
                print(f"the 2 have a similarity score of: {cosine_sim}")
                print("================================================================================")

                relevant_chunks.append(cosine_sim)
                break
            i += 1

print(relevant_chunks)


    