
input = {"context1Key": {
   "GPT": "I like cheese",
   "model1": "I like to eat cheese",
   "model7": "Cheesy food is my like I,"
 },
 "context2Key": {
   "GPT": "What is the answer to life",
   "model1": "The life is cool",
   "model7": "To answer life"
 }
}

import evaluate

def evaluate_contexts(input, evaluation_name):
  evaluation_metric = evaluate.load(evaluation_name)

  output = {}
  
  for context in input:
    reference = [input[context]["GPT"]]
    models = input[context]
    output[context] = {}

    for model in models:
      if model != "GPT":
        prediction = [models[model]]
        result = evaluation_metric.compute(predictions=prediction, references=reference)

        if evaluation_name == "google_blue":
          output[context][model] = result["google_bleu"]

        if evaluation_name == "rouge":
          output[context][model] = result["rougeL"]

        if evaluation_name == "meteor":
          output[context][model] = result["meteor"]
      
  return output

def evaluate_contexts_all_metrics(input):
  evaluation_google_blue = evaluate.load("google_bleu")
  evaluation_rouge = evaluate.load("rouge")
  evaluation_meteor = evaluate.load("meteor")

  output = {}
  
  for context in input:
    reference = [input[context]["GPT"]]
    models = input[context]
    output[context] = {}

    for model in models:
      if model != "GPT":
        prediction = [models[model]]
        result_google_bleu = evaluation_google_blue.compute(predictions=prediction, references=reference)
        result_rogue = evaluation_rouge.compute(predictions=prediction, references=reference)
        result_google_meteor = evaluation_meteor.compute(predictions=prediction, references=reference)

        output[context][model] = {}        
        output[context][model]["google_bleu"] = result_google_bleu["google_bleu"]
        output[context][model]["rouge"] = result_rogue["rougeL"]
        output[context][model]["meteor"] = result_google_meteor["meteor"]

  return output


input1 = {
  'context1Key': {
    'model1': {
      'google_blue': 0.2857142857142857, 
      'rouge': 0.7499999999999999, 
      'meteor': 0.7986111111111112
    }, 
    'model7': {
      'google_blue': 0.09090909090909091, 
      'rouge': 0.2222222222222222, 
      'meteor': 0.29411764705882354
    }
  },
  'context2Key': {
    'model1': {
      'google_blue': 0.2857142857142857, 
      'rouge': 0.7499999999999999, 
      'meteor': 0.7986111111111112
    }, 
    'model7': {
      'google_blue': 0.09090909090909091, 
      'rouge': 0.2222222222222222, 
      'meteor': 0.29411764705882354
    }
  }
}

def calculate_avg_evaluation(input):
  output = {}

  # just to initialise output structure
  for context in input:
    models = input[context]
    for model in models:
      output[model] = []
    
    break
  
  for context in input:
    models = input[context]
    for model in models:
      metrics = models[model]
      for metric in metrics:
        output[model].append(metrics[metric])
  
  for model in output:
    output[model] = sum(output[model]) / len(output[model])

calculate_avg_evaluation(input1)






