from transformers import HfArgumentParser, BartTokenizer, set_seed
from dataclasses import dataclass, field
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from angle_emb import AnglE
from datasets import load_dataset, concatenate_datasets

@dataclass
class GenerateArguments:
    max_token_limit: int = field(default="512", metadata={"help": "The max token limit for each chunk of the PMC article"})
    pretrained_bart_tokenizer_name: str = field(default="alinet/bart-base-balanced-qg", metadata={"help": "The name of the Bart Tokenizer model"})
    #change this line when using on cluster
    output_dir: str = field(default="./chromadb", metadata={"help": "The output dir for the vectordb"})
    collection_name: str = field(default="pubmedqa", metadata={"help": "The name of the collection in vectordb"})
    seed: int = field(default=42, metadata={"help": "Random seed"})


# Utility functions
def remove_until_last_fullstop(input_string):
    # Find the index of the last full stop
    last_fullstop_index = input_string.rfind('.')
    
    # If no full stop is found, return the original string
    if last_fullstop_index == -1:
        return input_string
    
    # Return the substring from the beginning of the string to the last full stop
    return input_string[:last_fullstop_index + 1]

def filter_and_combine_context(data):
  combined_context = ''
  
  for idx, context in enumerate(data['context']['contexts']):
    if idx == 0:
      combined_context += context
    else:
      combined_context += " " + context

  data['context'] = combined_context

  return data

def main():
    parser = HfArgumentParser((GenerateArguments,))
    args = parser.parse_args_into_dataclasses()[0]

    set_seed(args.seed)

    # Initialisations 
    tokenizer = BartTokenizer.from_pretrained(args.pretrained_bart_tokenizer_name)
    def len_tokenize(string):
      return len(tokenizer.tokenize(string))

    # Text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=args.max_token_limit,
        chunk_overlap=0,
        length_function=len_tokenize,
        is_separator_regex=False,
    )

    # ChromaDB client and collection
    client = chromadb.PersistentClient(path=args.output_dir)
    collection = client.get_or_create_collection(
      name=args.collection_name,
      metadata={"hnsw:space": "cosine"} # l2 is the default
    )

    # Embedding Model
    angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()

    pubmed_art_ds = (
      load_dataset("pubmed_qa", "pqa_artificial", split="train")
      .select_columns(["pubid", "question", "context", "long_answer"])
      .map(filter_and_combine_context)
    )
    pubmed_labeled_ds = (
      load_dataset("pubmed_qa", "pqa_labeled", split="train")
      .select_columns(["pubid", "question", "context", "long_answer"])
      .map(filter_and_combine_context)
    )
    pubmed_unlabeled_ds = (
      load_dataset("pubmed_qa", "pqa_unlabeled", split="train")
      .select_columns(["pubid", "question", "context", "long_answer"])
      .map(filter_and_combine_context)
    )   

    pubmed_ds = concatenate_datasets([pubmed_art_ds, pubmed_labeled_ds, pubmed_unlabeled_ds])

    for idx, data in enumerate(pubmed_ds):
      if idx == 10:
        break

      documents = text_splitter.create_documents([data['context']])

      chunk_id = ""
      for id, doc in enumerate(documents):
        filtered_page_content = remove_until_last_fullstop(doc.page_content)
        embedding = angle.encode(filtered_page_content, to_numpy=True)

        chunk_id = str(data['pubid']) + "C" + str(id)

        collection.add(
          embeddings=embedding,
          documents=filtered_page_content,
          ids=chunk_id, 
        )
    

if __name__ == "__main__":
    main()