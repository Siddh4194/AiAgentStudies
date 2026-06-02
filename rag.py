import os
from time import monotonic

from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain_ollama import OllamaEmbeddings, ChatOllama
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 1. PASTE YOUR MONGODB ATLAS CONNECTION URL HERE
# ============================================================
MONGODB_URI = os.getenv("MONGODB_URI")

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

# init chat model
chat_model = ChatOllama(
    model="llama3.2:3b",
    base_url="http://192.168.1.47:11434",
    temperature=0.1,
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
    
    if not results:
        print("No results found for the query.")
        return
    
    context_parts = []
    
    for index, (doc, score) in enumerate(results, start=1):
        print(f"\nScore: {score:.4f}")
        print("Text snippet:", doc.page_content[:200])
        context_parts.append(f"""
                             source {index}
                             page:{doc.metadata.get('page_number', 'N/A')}
                             Content: {doc.page_content}
                             """)
    context = "\n\n".join(context_parts)
    
    prompt = f"""You are a helpful assistant
    Answer only based on the provided context.
    If the answer is not in the context, say "The information is not in the document" no more explanation.
    
    Explain the answer in detail and provide step-by-step reasoning.
    At the end provide the page and source of the information.
    
    Context:{context}
    
    Question: {query}"""
    
    response = chat_model.invoke(prompt)
    
    print("\nResponse:\n", response)

query_data(
    "What is the milleage of the mercedez benz 2020 model?"
)