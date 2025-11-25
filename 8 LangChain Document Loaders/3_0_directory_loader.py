from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path = 'books',
    glob = '*.pdf',
    loader_cls = PyPDFLoader
)

docs = loader.load() #Takes time to load pdfs, and loads pdf into RAM. To solve this, we have LazyLoading option

print(len(docs)) # Total Pages
print(docs[0].page_content)
print(docs[0].metadata) #{'producer': 'Adobe PDF Library 10.0.1', 'creator': 'Adobe InDesign CS6 (Windows)', 'creationdate': '2015-03-24T13:14:02+05:30', 'moddate': '2015-03-25T17:33:08+05:30', 'trapped': '/False', 'source': 'books\\Book 1 Building Machine Learning Systems with Python - Second Edition.pdf', 'total_pages': 326, 'page': 0, 'page_label': 'Cover'}  

print('-'*75)

docs_lazy = loader.lazy_load() #Uses generator based to load each time on demand. Saves memory.

for doc in docs_lazy:
    print(doc.metadata)

print('-'*75)