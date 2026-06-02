import os
from pprint import pprint

from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_ollama import OllamaEmbeddings


load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")

if not MONGODB_URI:
    raise ValueError("MONGODB_URI is missing from .env")


DATABASE_NAME = "rag_database"
COLLECTION_NAME = "pdf_embeddings"
VECTOR_INDEX_NAME = "autoembed_index"


client = MongoClient(MONGODB_URI)
client.admin.command("ping")

print("[OK] Connected to MongoDB")

collection = client[DATABASE_NAME][COLLECTION_NAME]


# ------------------------------------------------------------
# Check stored documents
# ------------------------------------------------------------

count = collection.count_documents({})

print("\nStored documents:", count)

if count == 0:
    raise RuntimeError("Collection is empty")


sample = collection.find_one()

print("\nSample fields:")
print(sample.keys())

stored_embedding = sample.get("embedding")

if not stored_embedding:
    raise RuntimeError("Embedding field is missing")

print("Stored embedding dimensions:", len(stored_embedding))
print("Text preview:", sample.get("text", "")[:200])


# ------------------------------------------------------------
# Check Atlas Vector Search index
# ------------------------------------------------------------

print("\nSearch indexes:")

indexes = list(collection.list_search_indexes())

if not indexes:
    raise RuntimeError("No search index found on this collection")

for search_index in indexes:
    pprint(search_index)


# ------------------------------------------------------------
# Connect to Ollama
# ------------------------------------------------------------

embeddings_model = OllamaEmbeddings(
    model="nomic-embed-text:latest",
    base_url="http://192.168.1.47:11434",
)


query = (
    "Explain relational databases, tables, rows, columns, "
    "primary keys and foreign keys"
)

query_embedding = embeddings_model.embed_query(query)

print("\nQuery embedding dimensions:", len(query_embedding))


if len(query_embedding) != len(stored_embedding):
    raise RuntimeError(
        f"Dimension mismatch: stored={len(stored_embedding)}, "
        f"query={len(query_embedding)}"
    )


# ------------------------------------------------------------
# Test MongoDB directly
# ------------------------------------------------------------

pipeline = [
    {
        "$vectorSearch": {
            "index": VECTOR_INDEX_NAME,
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": 50,
            "limit": 3,
        }
    },
    {
        "$project": {
            "_id": 0,
            "text": 1,
            "score": {
                "$meta": "vectorSearchScore"
            },
        }
    },
]

raw_results = list(collection.aggregate(pipeline))

print("\nRaw MongoDB results:", len(raw_results))

for index, result in enumerate(raw_results, start=1):
    print(f"\n--- Raw Result {index} ---")
    print("Score:", result.get("score"))
    print(result.get("text", "")[:500])


# ------------------------------------------------------------
# Test LangChain only after raw MongoDB works
# ------------------------------------------------------------

vector_store = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string=MONGODB_URI,
    namespace=f"{DATABASE_NAME}.{COLLECTION_NAME}",
    embedding=embeddings_model,
    index_name=VECTOR_INDEX_NAME,
)

langchain_results = vector_store.similarity_search_with_score(
    query=query,
    k=3,
)

print("\nLangChain results:", len(langchain_results))

for index, (document, score) in enumerate(langchain_results, start=1):
    print(f"\n--- LangChain Result {index} ---")
    print("Score:", score)
    print(document.page_content[:500])