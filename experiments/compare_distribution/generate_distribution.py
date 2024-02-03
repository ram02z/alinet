from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset

model_name = "alinet/t5-base-squad-qg"
t5_base_tokenizer = T5Tokenizer.from_pretrained(model_name)
t5_base = T5ForConditionalGeneration.from_pretrained(model_name)

model_name = "alinet/t5-base-balanced-qg"
t5_balanced_tokenizer = T5Tokenizer.from_pretrained(model_name)
t5_balanced = T5ForConditionalGeneration.from_pretrained(model_name)

def generate_questions(data):
  source = data['source']
  
  baseline_question1b = run_model(source, t5_base, t5_base_tokenizer, max_length=32, num_beams=1)
  balanced_question1b = run_model(source, t5_balanced, t5_balanced_tokenizer, max_length=32, num_beams=1)

  baseline_question4b = run_model(source, t5_base, t5_base_tokenizer, max_length=32, num_beams=4)
  balanced_question4b = run_model(source, t5_balanced, t5_balanced_tokenizer, max_length=32, num_beams=4)

  data['baseline1b'] = baseline_question1b[0]
  data['balanced1b'] = balanced_question1b[0]

  data['baseline4b'] = baseline_question4b[0]
  data['balanced4b'] = balanced_question4b[0]

  return data

def contain_unique_question_context(data, unique_sources):
  if data['source'] in unique_sources:
    return False
  else:
    unique_sources.add(data['source'])
    return True
  
def run_model(input_string, model, tokenizer, **generator_args):
  input_ids = tokenizer.encode(input_string, return_tensors="pt")
  res = model.generate(input_ids, **generator_args)
  output = tokenizer.batch_decode(res, skip_special_tokens=True)
  return output

def main():
  compare_dataset = load_dataset("csv", data_files="../../data/validation.csv", split='train')

  unique_sources = set()
  compare_dataset = compare_dataset.filter(contain_unique_question_context, fn_kwargs={"unique_sources": unique_sources})

  compare_dataset = (
    compare_dataset
    .add_column("baseline1b", [None] * len(compare_dataset))
    .add_column("balanced1b", [None] * len(compare_dataset))
    .add_column("baseline4b", [None] * len(compare_dataset))
    .add_column("balanced4b", [None] * len(compare_dataset))
    .map(generate_questions)
  )

  compare_dataset.to_csv("compare.csv")

if __name__ == "__main__":
    main()