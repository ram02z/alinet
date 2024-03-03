from datasets import load_dataset, concatenate_datasets

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
  pubmed_dict_ds = pubmed_ds.train_test_split(test_size = 0.2)
  
  pubmed_train_ds = pubmed_dict_ds['train']
  pubmed_validation_ds = pubmed_dict_ds['test']

  pubmed_train_ds.to_csv("./data/pubmed_qa_train.csv")
  pubmed_validation_ds.to_csv("./data/pubmed_qa_validation.csv")


if __name__ == "__main__":
    main()