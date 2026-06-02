from time import monotonic

from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain_ollama import OllamaEmbeddings
# ============================================================
# 1. PASTE YOUR MONGODB ATLAS CONNECTION URL HERE
# ============================================================

MONGODB_URI = (
    "mongodb+srv://Siddh1418:<PASSWORD>@chatbot.4zqxkh3.mongodb.net/?appName=Chatbot"
)

DATABASE_NAME = "rag_database"
COLLECTION_NAME = "pdf_embeddings"
VECTOR_INDEX_NAME = "autoembed_index"

client = MongoClient(MONGODB_URI)

client.admin.command("ping")

print("Connected to MongoDB successfully")

# "fields": [

# "numDimensions": 1536,
# "path": "embedding",
# "similarity": "cosine",
# "type": "vector"
# 1,
# {

# "path": "hasCode",
# "type": "filter"

# init embeddings model
embeddings_model = OllamaEmbeddings(
    model="nomic-embed-text:latest",
    base_url="http://192.168.1.47:11434",
)


vectorStore = MongoDBAtlasVectorSearch.from_connection_string(
    MONGODB_URI,
    DATABASE_NAME + "." + COLLECTION_NAME,
    embeddings_model,
    index_name=VECTOR_INDEX_NAME,
)

def query_data(query: str):
    results = vectorStore.similarity_search_with_score(
        query=query,
        k=3,
    )

    print("Retrieved documents:", len(results))

    for index, (document, score) in enumerate(results, start=1):
        print(f"\n--- Result {index} ---")
        print("Score:", score)
        print(document.page_content)
        print("Metadata:", document.metadata)


query_data(
    "Primary Keys: In the relational model, each row in a table must have a unique" 
"identifier, which is known as the primary key. This ensures that each row is"
"unique, can be accessed, and manipulated easily. "
)