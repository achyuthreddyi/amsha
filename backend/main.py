# main.py

from fastapi import FastAPI, File, UploadFile
import os

app = FastAPI()
DOCUMENT_DIR = "document_store"

from utils import extract_text_from_file, chunk_text
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from pydantic import BaseModel
from openai import OpenAI
import requests

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ChromaDB client
chroma_client = chromadb.Client()

# Collection name
COLLECTION_NAME = "knowledge"
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)


# Add a dictionary to track processing status
processing_status = {}

os.makedirs(DOCUMENT_DIR, exist_ok=True)

@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    file_path = os.path.join(DOCUMENT_DIR, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    return {"status": "success", "filename": file.filename}


from fastapi import BackgroundTasks

@app.post("/process/{filename}")
def process_file(filename: str, background_tasks: BackgroundTasks):
    file_path = os.path.join(DOCUMENT_DIR, filename)

    if not os.path.exists(file_path):
        return {"error": "File not found"}

    # Initialize status as "processing"
    processing_status[filename] = "processing"
    
    # Run in background to avoid blocking
    # background_tasks.add_task(vectorize_and_store, file_path, filename)
    vectorize_and_store(file_path, filename)
    return {"status": "processing started", "filename": filename}

def vectorize_and_store(file_path: str, doc_id_prefix: str):
    try:
        text = extract_text_from_file(file_path)
        print('Printing the text of the files', text)
        chunks = chunk_text(text)
        print('Printing the chunks of the files', chunks)
        embeddings = embedding_model.encode(chunks).tolist()

        print('Printing the embeddings of the files', embeddings)
        ids = [f"{doc_id_prefix}_{i}" for i in range(len(chunks))]
        print('Printing the ids of the files', ids)
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids
        )
        
        # Update status to "completed" with additional info
        processing_status[doc_id_prefix] = {
            "status": "completed",
            "chunks": len(chunks),
            "vectors": len(embeddings)
        }
    except Exception as e:
        # Update status to "failed" with error message
        processing_status[doc_id_prefix] = {
            "status": "failed",
            "error": str(e)
        }

# Add a new endpoint to check processing status
@app.get("/status/{filename}")
def get_processing_status(filename: str):
    if filename in processing_status:
        return {"filename": filename, "status": processing_status[filename]}
    else:
        return {"filename": filename, "status": "not found"}

@app.get("/list_documents")
def list_documents():
    # This will return all document IDs in the Chroma collection
    return {
        "ids": collection.get()["ids"],
        "count": len(collection.get()["ids"])
    }

@app.get("/preview_vectors/{doc_id_prefix}")
def preview_vectors(doc_id_prefix: str):
    results = collection.get()
    preview = []

    print('Printing the results of the preview', len(results))
    print('Printing the results of the preview', results)

    for id_, doc in zip(results["ids"], results["documents"]):
        if id_.startswith(doc_id_prefix):
            preview.append({"id": id_, "text": doc})
    
    return {"preview": preview}

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3  # How many chunks to retrieve

@app.post("/query")
def query_docs(request: QueryRequest):
    query_embedding = embedding_model.encode([request.question])[0]
    # print('Printing the query embedding', query_embedding)
    results = collection.query(query_embeddings=[query_embedding], n_results=request.top_k)
    print('Printing the results of the query', results)
    top_chunks = results['documents'][0]
    print('Printing the top chunks for sending to the LLM', top_chunks)

    prompt = f"""You are a helpful assistant. Use the following context to answer the question.

Context:
{chr(10).join(top_chunks)}

Question: {request.question}
Answer:"""

    # 🧠 Call local LLM using Ollama
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }
    )

    output = response.json()
    print("🔍 Raw Ollama response:", output)
    return {
        "answer": output["response"]
    }