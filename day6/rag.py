import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

# --- Step 1: Load and chunk the PDF ---
loader = PyPDFLoader("company_handbook.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(pages)
chunk_texts = [chunk.page_content for chunk in chunks]

print(f"Loaded and split into {len(chunk_texts)} chunks\n")

# --- Step 2: Embed chunks and store in ChromaDB ---
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.create_collection(name="handbook")

embeddings = model.encode(chunk_texts).tolist()

collection.add(
    embeddings=embeddings,
    documents=chunk_texts,
    ids=[f"chunk_{i}" for i in range(len(chunk_texts))]
)

# --- Step 3: Set up the LLM chain ---
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")

prompt = ChatPromptTemplate.from_messages([
    ("system", """Answer using ONLY the context below. If the answer isn't in the context, say "I don't have information about that in the document."

Context:
{context}"""),
    ("user", "{question}")
])

chain = prompt | llm

# --- Step 4: Ask questions ---
print("Handbook RAG ready. Type 'quit' to exit.\n")

while True:
    question = input("Question: ")
    if question.lower() == "quit":
        break

    question_embedding = model.encode([question]).tolist()
    results = collection.query(query_embeddings=question_embedding, n_results=3)

    context = "\n".join(results["documents"][0])
    response = chain.invoke({"context": context, "question": question})

    print(f"\nAnswer: {response.content}\n")
    print("--- Retrieved chunks used ---")
    for doc in results["documents"][0]:
        print("-", doc[:80], "...")
    print()