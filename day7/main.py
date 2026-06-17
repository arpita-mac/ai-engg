from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Handbook RAG Assistant")

# --- Setup runs ONCE when the server starts, not on every request ---

print("Loading and chunking PDF...")
loader = PyPDFLoader("company_handbook.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(pages)
chunk_texts = [chunk.page_content for chunk in chunks]

print("Embedding chunks and storing in ChromaDB...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="handbook")

embeddings = embed_model.encode(chunk_texts).tolist()
collection.add(
    embeddings=embeddings,
    documents=chunk_texts,
    ids=[f"chunk_{i}" for i in range(len(chunk_texts))]
)

llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")

prompt = ChatPromptTemplate.from_messages([
    ("system", """Answer using ONLY the context below. If the answer isn't in the context, say "I don't have information about that in the document."

Context:
{context}"""),
    ("user", "{question}")
])

chain = prompt | llm

print("Ready.")

# --- Request/response models ---

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: list[str]

# --- Routes ---

@app.get("/")
def root():
    return {"status": "Handbook RAG Assistant is running"}

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    question_embedding = embed_model.encode([request.question]).tolist()
    results = collection.query(query_embeddings=question_embedding, n_results=3)

    retrieved_chunks = results["documents"][0]
    context = "\n".join(retrieved_chunks)

    response = chain.invoke({"context": context, "question": request.question})

    return AskResponse(
        answer=response.content,
        sources=retrieved_chunks
    )