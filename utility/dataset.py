def print_distribution(dataset):
  method_ds = dataset.filter(lambda data: data["category"] == "method")
  description_ds = dataset.filter(lambda data: data["category"] == "description")
  explanation_ds = dataset.filter(lambda data: data["category"] == "explanation")
  comparison_ds = dataset.filter(lambda data: data["category"] == "comparison")
  recall_ds = dataset.filter(lambda data: data["category"] == "recall")

  na_ds = dataset.filter(lambda data: data["category"] == "NA")

  print("description distribution =" + str( len(description_ds) / len(dataset) * 100) + "%, count = " + str(len(description_ds)))
  print("recall distribution = " + str( len(recall_ds) / len(dataset) * 100) + "%, count = " + str(len(recall_ds)))
  print("explanation distribution = " + str( len(explanation_ds) / len(dataset) * 100) + "%, count = " + str(len(explanation_ds)))
  print("method distribution = " + str( len(method_ds) / len(dataset) * 100) + "%, count = " + str(len(method_ds)))
  print("comparison distribution = " + str( len(comparison_ds) / len(dataset) * 100) + "%, count = " + str(len(comparison_ds)))
  print("na distribution = " + str( len(na_ds) / len(dataset) * 100) + "%, count = " + str(len(na_ds)))