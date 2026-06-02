from pymongo import MongoClient

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch


# ============================================================
# 1. PASTE YOUR MONGODB ATLAS CONNECTION URL HERE
# ============================================================

MONGODB_URI = (
    "mongodb+srv://Siddh1418:<PASSWORD>@chatbot.4zqxkh3.mongodb.net/?appName=Chatbot"
)

DATABASE_NAME = "rag_database"
COLLECTION_NAME = "pdf_embeddings"
VECTOR_INDEX_NAME = "vector_index"


# ============================================================
# 2. CONNECT TO MONGODB
# ============================================================

client = MongoClient(MONGODB_URI)

# Verify the connection before generating embeddings
client.admin.command("ping")
print("Connected to MongoDB successfully")

database = client[DATABASE_NAME]
collection = database[COLLECTION_NAME]


# ============================================================
# 3. LOAD AND SPLIT THE PDF
# ============================================================

loader = PyPDFLoader(
    r"assets\5723a07ce6db1b2e7fe33b5db5f0d606.pdf"
)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)

splits = text_splitter.split_documents(documents)

print(f"Generated {len(splits)} chunks")
print("\nSample chunk:\n")
print(splits[1].page_content)


# ============================================================
# 4. LOAD THE LOCAL OLLAMA EMBEDDING MODEL
# ============================================================

embeddings_model = OllamaEmbeddings(
    model="nomic-embed-text:latest",
    base_url="http://192.168.1.47:11434",
)


# ============================================================
# 5. CONNECT LANGCHAIN TO THE MONGODB COLLECTION
# ============================================================

vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings_model,
    index_name=VECTOR_INDEX_NAME,
)


# ============================================================
# 6. GENERATE EMBEDDINGS AND STORE THE CHUNKS IN MONGODB
# ============================================================

inserted_ids = vector_store.add_documents(splits)

print(f"\nInserted {len(inserted_ids)} chunks into MongoDB")
print(f"Stored documents: {collection.count_documents({})}")