from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='social_netwrok_ads.csv')

docs = loader.load()

print(len(docs)) #Total rows
print('-'*75)
print(docs[0])