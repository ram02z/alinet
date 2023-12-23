import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd

model_name = 't5-base'
train_file_path = 'datasets/squad_train.csv'
validation_file_path = 'datasets/squad_validation.csv'
save_model_path = 'model/'
save_tokenizer_path = 'tokenizer/'
pretrained_model = 't5-base'

max_length_input = 512 
max_length_output = 128


if __name__ == "__main__":
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print('Using device:', device)

  # Load PLM
  model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
  tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=True)

  # load dataset
  train_data = pd.read_csv(train_file_path)

  # initialise inpus and targets

  input_ids = []
  attention_masks = []
  labels = []

  for index, row in train_data.iterrows():
    # input encoding
    encoding = tokenizer(
      [row["context"]],
      padding="longest",
      max_length=max_length_input,
      truncation=True,
      return_tensors="pt",
    )
    # append encoded context input and attention_mask
    input_ids.append(encoding.input_ids[0])
    attention_masks.append(encoding.attention_mask[0])

    # target encoding
    target_encoding = tokenizer(
      [row["question"]],
      padding="longest",
      max_length=max_length_output,
      truncation=True,
      return_tensors="pt",
    )
    # replace padding token id's of the labels by -100 so it's ignored by the loss
    target_encoding.input_ids[0][target_encoding.input_ids[0] == tokenizer.pad_token_id] = -100
    labels.append(target_encoding.input_ids[0])


    if index == 2:
      break

  print(input_ids)
  print(attention_masks)
  print(labels)
    
  
  


