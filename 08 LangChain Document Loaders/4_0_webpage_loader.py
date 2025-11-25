from langchain_community.document_loaders import WebBaseLoader

url = 'https://docs.langchain.com/oss/python/integrations/document_loaders'
loader = WebBaseLoader(url)

docs = loader.load()

print(len(docs))
print('-'*75)
print(docs)

#Can create a sample project like chrome plugin to chat with webpage we are looking at