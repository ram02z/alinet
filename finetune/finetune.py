import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import pandas as pd
from datasets import Dataset
import os

model_name = 't5-base'
train_file_path = 'datasets/squad_train.csv'
validation_file_path = 'datasets/squad_validation.csv'
save_model_path = 'model'
pretrained_model = 't5-base'

max_length_input = 512 
max_length_output = 128


def generate_encodings(dataset):
    inputs = [context for context in dataset["context"]]
    targets = [question for question in dataset["question"]]
    input_encoding = tokenizer( 
      inputs,
      padding="longest",
      max_length=max_length_input,
      truncation=True,
      return_tensors="pt",
    )

    # Setup the tokenizer for targets
    target_encoding = tokenizer(
      targets,
      padding="longest",
      max_length=max_length_output,
      truncation=True,
      return_tensors="pt",
    )

    labels = target_encoding["input_ids"]
    # replace padding token id's of the labels by -100 so it's ignored by the loss
    labels[labels == tokenizer.pad_token_id] = -100

    # create new column called labels and assigns labels to it
    input_encoding["labels"] = labels 

   
    return input_encoding

if __name__ == "__main__":
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Load PLM
  model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
  tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=True)

  # load dataset
  train_data = pd.read_csv(train_file_path)
  validation_data = pd.read_csv(validation_file_path)

  train_dataset = Dataset.from_pandas(train_data)
  validation_dataset = Dataset.from_pandas(train_data)

  
  tokenized_train_dataset = train_dataset.map(generate_encodings, batched=True)
  tokenized_eval_dataset = validation_dataset.map(generate_encodings, batched=True)
  
  training_args = TrainingArguments(
    output_dir='model_res_out',
    overwrite_output_dir=True,
    num_train_epochs=1,
    weight_decay=0.01,
    save_total_limit=5,
    learning_rate=0.0001,
    per_device_train_batch_size = 5
  )
 
  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
  )

  # Train the model
  trainer.train()

  if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)

  model.save_pretrained(save_model_path)
  
    
      

# initialise inpus and targets
  # for index, row in train_data.iterrows():
  #   # input encoding
  #   encoding = tokenizer(
  #     [row["context"]],
  #     padding="longest",
  #     max_length=max_length_input,
  #     truncation=True,
  #     return_tensors="pt",
  #   )
  #   input_ids = encoding.input_ids
  #   attention_mask = encoding.attention_mask
  
  #   # target encoding
  #   target_encoding = tokenizer(
  #     [row["question"]],
  #     padding="longest",
  #     max_length=max_length_output,
  #     truncation=True,
  #     return_tensors="pt",
  #   )
  #   labels = target_encoding.input_ids
  #   # replace padding token id's of the labels by -100 so it's ignored by the loss
  #   labels[labels == tokenizer.pad_token_id] = -100

  #   loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
  #   print(loss.item())
  #   break
  
  


