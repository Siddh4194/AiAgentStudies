import os
from uuid import uuid4

from dotenv import load_dotenv
from pymongo import MongoClient

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings

from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.store.mongodb import MongoDBStore, create_vector_index_config
from langgraph.utils.config import get_config


load_dotenv()

# ---------------------------------------------------------
# MongoDB connection
# ---------------------------------------------------------

mongodb_uri = os.getenv("MONGODB_URI")

if not mongodb_uri:
    raise ValueError("MONGODB_URI is missing from the .env file.")

client = MongoClient(mongodb_uri)
client.admin.command("ping")

print("Connected to MongoDB successfully")


# ---------------------------------------------------------
# Embedding model
# ---------------------------------------------------------

embeddings_model = OllamaEmbeddings(
    model="nomic-embed-text:latest",
    base_url="http://192.168.1.47:11434",
)


# ---------------------------------------------------------
# Long-term memory store
# ---------------------------------------------------------

memory_collection = client["agent_memory"]["memories"]

index_config = create_vector_index_config(
    embed=embeddings_model,
    dims=768,
    relevance_score_fn="dotProduct",
    fields=["content"],
)

store = MongoDBStore(
    memory_collection,
    index_config=index_config,
    auto_index_timeout=120,
)


# ---------------------------------------------------------
# Chat model
# ---------------------------------------------------------

chat_model = ChatOllama(
    model="llama3.2:3b",
    base_url="http://192.168.1.47:11434",
    temperature=0.1,
)


# ---------------------------------------------------------
# Long-term memory tools
# ---------------------------------------------------------

@tool
def store_memory(content: str) -> str:
    """Store a long-term memory for the current user."""

    config = get_config()
    user_id = config.get("configurable", {}).get("user_id")

    if not user_id:
        return "Unable to store memory because user_id is missing."

    # User-level isolation for long-term memories
    namespace = ("user", user_id, "memories")

    store.put(
        namespace=namespace,
        key=str(uuid4()),
        value={"content": content},
    )

    return f"Memory stored successfully: {content}"


@tool
def retrieve_memories(query: str, top_k: int = 5) -> str:
    """Retrieve relevant long-term memories for the current user."""

    config = get_config()
    user_id = config.get("configurable", {}).get("user_id")

    if not user_id:
        return "Unable to retrieve memories because user_id is missing."

    # Must match store_memory() exactly
    namespace = ("user", user_id, "memories")

    results = store.search(
        namespace,
        query=query,
        limit=top_k,
    )

    if not results:
        return "No relevant memories found."

    memories = [result.value["content"] for result in results]

    return "Relevant memories:\n" + "\n".join(memories)


# ---------------------------------------------------------
# Agent
# ---------------------------------------------------------

system_prompt = """
You are a helpful AI assistant with memory capabilities.

When the user sends a message:
1. First, call retrieve_memories to check whether relevant user memories exist.
2. Use relevant memories to personalize the answer.
3. If the user shares a useful new personal fact or preference, call store_memory.
4. Do not claim that you remember information unless it exists in memory.
"""

checkpointer = MongoDBSaver(client)

agent = create_agent(
    model=chat_model,
    system_prompt=system_prompt,
    tools=[store_memory, retrieve_memories],
    checkpointer=checkpointer,
    store=store,
)


# ---------------------------------------------------------
# Chat function
# ---------------------------------------------------------

def chat(user_id: str, conversation_id: str, query: str) -> None:
    # IMPORTANT FIX:
    # The checkpoint thread must be unique for each user and conversation.
    thread_id = f"user:{user_id}:thread:{conversation_id}"

    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
        }
    }

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ]
        },
        config=config,
    )

    print(f"User ID: {user_id}")
    print(f"Thread ID: {thread_id}")
    print(f"User: {query}")
    print("Response:", response["messages"][-1].content)
    print("-" * 60)


# ---------------------------------------------------------
# Isolation test
# ---------------------------------------------------------

chat("Siddhant", "thread1", "My name is Siddhant. Please remember it.")
chat("Sanskruti", "thread1", "My name is Sanskruti. Please remember it.")

# New conversations: must use long-term memories rather than chat history
chat("Siddhant", "thread2", "What is my name?")
chat("Sanskruti", "thread2", "What is my name?")