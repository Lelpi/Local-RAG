from dotenv import load_dotenv
import os
import json
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4
import chromadb


load_dotenv()

if os.getenv("USE_BRIGHTDATA", "False") == "True":
    collection_name = os.getenv("COLLECTION_NAME_BRIGHTDATA")
    dataset_file = os.getenv("DATASET_STORAGE_FILE_BRIGHTDATA")
else:
    collection_name = os.getenv("COLLECTION_NAME_FREE")
    dataset_file = os.getenv("DATASET_STORAGE_FILE_FREE")

# Delete only the target collection so other collections are preserved
client = chromadb.PersistentClient(path=os.getenv("DATABASE_LOCATION"))
if collection_name in [c.name for c in client.list_collections()]:
    client.delete_collection(collection_name)
    print(f"Deleted existing collection '{collection_name}'")

# Initialize Chroma vector store
vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL")),
    persist_directory=os.getenv("DATABASE_LOCATION"), 
)

# Load the scraped data from the datasets directory. Each line in the dataset file is a JSON object with the article URL, title, raw text, and other fields
file_content = []
with open(f"{os.getenv('DATASET_STORAGE_FOLDER')}/{dataset_file}", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        file_content.append(json.loads(line))

# Initialize the text splitter with a chunk size of 1000 characters and an overlap of 200 characters between chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

# Iterate through the scraped articles, split the raw text into chunks, and add the chunks to the Chroma vector store with a unique ID and metadata containing the source URL and title of the article
for article in file_content:
    print(article['url'])
    texts = text_splitter.create_documents([article['raw_text']], metadatas=[{"source": article['url'], "title": article['title']}])
    uuids = [str(uuid4()) for _ in range(len(texts))]
    vector_store.add_documents(documents=texts, ids=uuids)
