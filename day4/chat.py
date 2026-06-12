from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not set.")

llm = ChatGroq(
    api_key=api_key,
    model="llama-3.3-70b-versatile"
)

print("Chat started. Type 'quit' to exit.\n")
history = [
    SystemMessage(content="You are a helpful assistant.")
]

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        break
    if not user_input.strip():
        continue

    history.append(HumanMessage(content=user_input))

    try:
        response = llm.invoke(history)
        print(f"AI: {response.content}\n")

        history.append(AIMessage(content=response.content))

    except Exception as e:
        print(f"Error: {e}\n")