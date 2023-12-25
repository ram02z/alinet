from datasets import load_dataset, concatenate_datasets
import pandas as pd
from sklearn.model_selection import train_test_split
import os


dataset_save_path = 'datasets/'

# if false, it removes sample
def contain_question_mark(data):
  return "?" in data["target"] 


# during filter for using context
# if current context is in unique_contexts sets
  # then return false
# else
  # return true

def contain_unique_question_context(data, unique_sources):
  if data['source'] in unique_sources:
    return False
  else:
    unique_sources.add(data['source'])
    return True


def load_squad_dataset():
  dataset = load_dataset("squad", split="train+validation")  

  dataset = dataset.select_columns(['context', 'question'])
  dataset = dataset.rename_columns(
    {"context": "source", "question": "target"}
  )

  unique_sources = set()
  dataset = dataset.filter(contain_question_mark)
  dataset = dataset.filter(contain_unique_question_context, fn_kwargs={"unique_sources": unique_sources})
  print(len(dataset["source"]))

  return dataset




if __name__ == "__main__":
  df_list = []

  df_list.append(load_squad_dataset())
  


 
  # dataset_save_path = 'datasets/'
  # if not os.path.exists(dataset_save_path):
  #   os.makedirs(dataset_save_path)
  # combined_df.to_csv(dataset_save_path + 'dataset.csv', index=False)


