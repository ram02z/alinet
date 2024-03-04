import chromadb


def main():
  print("loading vectordb")
  client = chromadb.PersistentClient(path="./chromadb")
  collection = client.get_collection("pubmedqa_validation")
  
  print(collection.peek())
  print(collection.count())

if __name__ == "__main__":
    main()