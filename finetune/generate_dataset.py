from datasets import load_dataset, concatenate_datasets
import pandas as pd
from sklearn.model_selection import train_test_split
import os


dataset_save_path = 'datasets/'

def load_squad_dataset():
  df_dataset = pd.DataFrame(columns=['dataset', 'context', 'question'])
  dataset = load_dataset("squad")  

  # combined = train and validation which includes test
  combined_dataset = concatenate_datasets([dataset["train"], dataset["validation"]])

  for idx in range(0, len(combined_dataset['question'])):
      context = combined_dataset['context'][idx]
      question = combined_dataset['question'][idx]

      data = {'dataset': 'squad', 'context': context, 'question': question}
      df_dataset.loc[idx] = data
    
  return df_dataset

def load_race_dataset():
  df_dataset = pd.DataFrame(columns=['dataset', 'context', 'question'])
  dataset = load_dataset("race", "all")  

  # combined = train and validation which includes test
  combined_dataset = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])
  
  for idx in range(0, len(combined_dataset['question'])):
      context = combined_dataset['article'][idx]
      question = combined_dataset['question'][idx]

      # skip questions that does not contain ?, if it takes too long can do question[:-1]
      if '?' not in question:
         continue

      data = {'dataset': 'race', 'context': context, 'question': question}
      df_dataset.loc[idx] = data
  
  return df_dataset

def load_sciq_dataset():
  df_dataset = pd.DataFrame(columns=['dataset', 'context', 'question'])
  dataset = load_dataset("sciq")  

  # combined = train and validation which includes test
  combined_dataset = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])

  for idx in range(0, len(combined_dataset['question'])):
      context = combined_dataset['support'][idx]
      question = combined_dataset['question'][idx]

      data = {'dataset': 'sciq', 'context': context, 'question': question}
      df_dataset.loc[idx] = data
  
  return df_dataset

def load_yahoo_dataset():
  df_dataset = pd.DataFrame(columns=['dataset', 'context', 'question'])
  # only contain train split
  dataset = load_dataset("yahoo_answers_qa", split="train")

  for idx in range(0, len(dataset['question'])):
      context = dataset['answer'][idx]
      question = dataset['question'][idx]

      data = {'dataset': 'yahoo', 'context': context, 'question': question}
      df_dataset.loc[idx] = data
  
  return df_dataset

if __name__ == "__main__":
  df_list = []

  df_list.append(load_squad_dataset())
  df_list.append(load_race_dataset())
  df_list.append(load_sciq_dataset())
  df_list.append(load_yahoo_dataset())

  # Concatenate the all datasets
  combined_df = pd.concat(df_list, ignore_index=True)

  dataset_save_path = 'datasets/'
  if not os.path.exists(dataset_save_path):
    os.makedirs(dataset_save_path)
  combined_df.to_csv(dataset_save_path + 'dataset.csv', index=False)


