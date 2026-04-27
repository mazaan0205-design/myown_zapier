import os
import io
from fastapi import FastAPI, Form, UploadFile, File
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# Database Setup
db_path = os.path.join(os.getcwd(), "chroma_db")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)

@app.get("/")
async def root():
    return {"status": "W3S API Running", "db": "ChromaDB Connected"}

# ENDPOINT 1: Ingest Data
@app.post("/ingest")
async def ingest_knowledge(file: UploadFile = File(...)):
    content = ""
    if file.filename.endswith(".pdf"):
        pdf_data = await file.read()
        pdf_reader = PdfReader(io.BytesIO(pdf_data))
        for page in pdf_reader.pages:
            content += page.extract_text()
    else:
        content = (await file.read()).decode("utf-8")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_text(content)
    vector_db.add_texts(chunks)
    
    return {"status": "Success", "added_chunks": len(chunks)}

# ENDPOINT 2: Chat
@app.post("/chat")
async def chat_endpoint(user_query: str = Form(...), instructions: str = Form(...)):
    # Retrieve top 3 chunks
    docs = vector_db.similarity_search(user_query, k=3)
    context = "\n---\n".join([d.page_content for d in docs])

    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
    
    prompt = f"ROLE: {instructions}\n\nKNOWLEDGE BASE:\n{context}"
    response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=user_query)])
    
    return {"response": response.content}