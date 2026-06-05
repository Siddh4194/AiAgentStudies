import os

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()


client = MongoClient(os.getenv("MONGODB_URI"))

client.admin.command("ping")

print("Connected to MongoDB successfully")

chat_model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.1,
    api_key=os.getenv("GROQ_API_KEY"),
) if False else  ChatOllama(
    model="llama3.2:3b",
    base_url="http://192.168.1.47:11434",
    temperature=0.1,
)




# init checkpointer

checkpointer = MongoDBSaver(client)


system_prompt = """You are a helpful assistant that provides concise answers to user queries."""

agent = create_agent(
    chat_model,
    system_prompt=system_prompt,
    tools=[],
    checkpointer=checkpointer
    )




def chat(user_id,thread_id, query):
    # thread isolation
    config = {"configurable":{"thread_id": thread_id,"user_id": user_id}}
    response = agent.invoke({"messages":[{"role": "user", "content": query}]}, config=config)
    print("Response:", response["messages"][-1].content)
    
# chat("Siddhant","thread2","What is the capital of France?")
# chat("Siddhant","thread2","I'm learning about the short term memory in the ai models")
# chat("Siddhant","thread2","I'm Siddhant Kadam")
# chat("Siddhant","thread2","What is my name")
    