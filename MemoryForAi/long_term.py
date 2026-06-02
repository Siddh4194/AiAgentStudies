import os

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.store.mongodb import MongoDBStore,create_vector_index_config
from langchain_core.tools import tool
from langgraph.utils.config import get_config
from pymongo import MongoClient
from langchain.agents import create_agent
from dotenv import load_dotenv
load_dotenv()

# init embeddings model
embeddings_model = OllamaEmbeddings(
    model="nomic-embed-text:latest",
    base_url="http://192.168.1.47:11434",
)

client = MongoClient(os.getenv("MONGODB_URI"))

client.admin.command("ping")

collection = client["agent_memory"]["memories"]

index_config = create_vector_index_config(
    embed=embeddings_model,
    dims=768,
    relevance_score_fn="dotProduct",
    fields=["content"]
)


store = MongoDBStore(collection,index_config=index_config,auto_index_timeout=120 )

print("Connected to MongoDB successfully")


# model
chat_model = ChatOllama(
    model="llama3.2:3b",
    base_url="http://192.168.1.47:11434",
    temperature=0.1,
)


@tool
def store_memory(content: str):
    """Store a memory in the database."""
    config = get_config()
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    store.put(namespace=("user", user_id), key=f"memory_{hash(content)}", value={"content": content})
    return "Memory stored : " + content

@tool
def retrieve_memories(query: str, top_k: int = 5):
    """Retrieve relevant memories from the database."""
    config = get_config()
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    namespace = ("user", user_id,"memories")
    results = store.search(namespace, query=query, top_k=top_k)
    if results:
        memories= [result.value["content"] for result in results]
        return "Relevant memories: " + "\n".join(memories)
    else:
        return "No relevant memories found."
    
system_prompt = """You are a helpful AI assistant with memory capabilities.
When a user sends you a message:
1. First, check your memory about them using retrieve_memories
2. Use what you find to personalize your response
3. If they share new information, save it using save_memory
Your memory persists across conversations!"""

agent = create_agent(chat_model, system_prompt=system_prompt, tools=[store_memory, retrieve_memories], checkpointer=MongoDBSaver(client))





def chat(user_id,thread_id, query):
    # thread isolation
    config = {"configurable":{"thread_id": thread_id,"user_id": user_id}}
    response = agent.invoke({"messages":[{"role": "user", "content": query}]}, config=config)
    print("Response:", response["messages"][-1].content)

# chat("Siddhant","thread3","What is the capital of France?")
# chat("Siddhant","thread3","My name is siddhant")
chat("Siddhant","thread4","What is my name?")