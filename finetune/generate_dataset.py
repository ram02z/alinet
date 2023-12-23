from datasets import load_dataset
import pandas as pd
from sklearn.utils import shuffle
import os

def preprocess_dataset(dataset):
  df_dataset = pd.DataFrame(columns=['context', 'question'])

  for idx in range(0, len(dataset['id'])):
      context = dataset['context'][idx]
      question = dataset['question'][idx]

      df_dataset.loc[idx] = [context] + [question] 
    
  return df_dataset

if __name__ == "__main__":
  train_dataset = load_dataset("squad", split='train')
  validate_dataset = load_dataset("squad", split='validation')


  # print(validate_dataset)

  df_train = preprocess_dataset(train_dataset)
  df_validate = preprocess_dataset(validate_dataset)

  # shuffling helps with training process, not sure if necessary
  df_train = shuffle(df_train)
  df_validate = shuffle(df_validate)

  #print('train: ', len(df_train))
  #print('validation: ', len(df_validate))

  dataset_save_path = 'datasets/'
  if not os.path.exists(dataset_save_path):
      os.makedirs(dataset_save_path)
  df_train.to_csv(dataset_save_path + 'squad_train.csv', index=False)
  df_validate.to_csv(dataset_save_path + 'squad_validation.csv', index=False)

# task, generate sum dataset in csv file
# split dataset in train val test split 80, 10, 10
# preprocess so that it only contains ? 
