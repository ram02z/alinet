from dataclasses import dataclass, field
from transformers import HfArgumentParser, AutoModelForSeq2SeqLM, AutoTokenizer, set_seed, BartForConditionalGeneration, BartTokenizer
from datasets import load_dataset, concatenate_datasets
from angle_emb import AnglE
import chromadb
from evaluate import Text2TextGenerationEvaluator
import evaluate



@dataclass
class GenerateArguments:
  seed: int = field(default=42, metadata={"help": "Random seed"})
  vectordb_path: str = field(default="./chromadb", metadata={"help": "The path to the vectordb"})
  collection_name: str = field(default="pubmedqa_validation", metadata={"help": "The name of the collection in the vectordb"})
  pretrained_bart_model_name: str = field(default="alinet/bart-base-balanced-qg", metadata={"help": "The name of the pretrained model"})
  num_beams: int = field(
    default=4,
    metadata={"help": "Number of beams to use for evaluation"},
  )
  seed: int = field(default=42, metadata={"help": "Random seed"})
  evaluation_module_id: str = field(
    default="bertscore",
    metadata={"help": "Id of the evaluation module from the HuggingFace hub"},
  )

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

    print("loading embedding model")
    angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()

    print("loading vectordb")
    client = chromadb.PersistentClient(path=args.vectordb_path)
    collection = client.get_collection(args.collection_name)

    def add_relevant_context_to_long_answer(data):
      query_embedding = angle.encode(data['long_answer'], to_numpy=True)

      relevant_query = collection.query(query_embeddings = query_embedding, n_results=1)

      relevant_context = relevant_query['documents'][0][0]

      long_answer_with_relevant_context = data['long_answer'] + " " + relevant_context 

      data['long_answer'] = long_answer_with_relevant_context

      return data

    print("loading dataset")
    eval_data = (
      load_dataset("alinet/pubmed_qa", split="validation")
      .map(add_relevant_context_to_long_answer)
      .select_columns(["long_answer", "question"])
      .rename_columns({"long_answer": "source", "question": "target"})
    )
    
    print("loading model")
    model = BartForConditionalGeneration.from_pretrained(
      args.pretrained_bart_model_name
    )
    tokenizer = BartTokenizer.from_pretrained(args.pretrained_bart_model_name)

    print("loading metric")
    metric = evaluate.load(args.evaluation_model_id)

    print("evaluating model")
    evaluator = Text2TextGenerationEvaluator()
    results = evaluator.compute(
      model_or_pipeline=model,
      tokenizer=tokenizer,
      data=eval_data,
      metric=metric,
      input_column="source",
      label_column="target",
      random_state=args.seed,
      generation_kwargs={
        "max_new_tokens": args.max_length,
        "num_beams": args.num_beams,
      },
    )

    print("saving results")
    evaluate.save("./results/", **results)

if __name__ == "__main__":
    main()