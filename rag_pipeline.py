import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def format_time(seconds):
    """Seconds ko MM:SS format mein convert karne ke liye"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def get_chroma_collection(db_path="vector_db"):
    """ChromaDB persistent client aur collection initialize karta hai"""
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name="video_lectures")
    return collection

def store_chunks_in_chroma(json_path, db_path="vector_db"):
    """Chunks ko load karke ChromaDB vector store mein embeddings ke sath save karta hai"""
    collection = get_chroma_collection(db_path)
    
    # Check if data already exists to avoid re-embedding
    if collection.count() > 0:
        print("[INFO] ChromaDB mein data pehle se stored hai. Skipping embedding generation.")
        return collection

    print("[INFO] Transcript chunks JSON file se load kiye ja rahe hain...")
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    ids, documents, embeddings, metadatas = [], [], [], []

    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk_{i}"
        text = chunk["text"]
        start = chunk["start"]
        end = chunk["end"]

        emb = model.encode(text).tolist()

        ids.append(chunk_id)
        documents.append(text)
        embeddings.append(emb)
        metadatas.append({"start": float(start), "end": float(end)})

    print("[INFO] Chunks aur embeddings ChromaDB mein store kiye ja rahe hain...")
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    print(f"[SUCCESS] Data successfully stored in ChromaDB vector database!")
    return collection

def retrieve_top_chunks(query, collection, top_k=5):  # <-- top_k ko 3 se 5 kar diya
    print(f"\n[QUERY] Student Question: '{query}'")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    top_chunks = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        top_chunks.append({
            "text": doc,
            "start": meta["start"],
            "end": meta["end"]
        })
    return top_chunks

def generate_ai_tutor_response(query, top_chunks):
    """Groq LLM ka use karke professional AI Teaching Assistant response generate karta hai"""
    print("[INFO] AI Teaching Assistant ke liye professional prompt banaya ja raha hai...")
    
    context_lines = []
    for chunk in top_chunks:
        start_fmt = format_time(chunk['start'])
        end_fmt = format_time(chunk['end'])
        context_lines.append(f"[Timestamp: {start_fmt} - {end_fmt}] {chunk['text']}")
    context = "\n".join(context_lines)
    
    prompt = f"""
    [ROLE]
    You are an expert, patient, and knowledgeable AI Teaching Assistant. Your goal is to help students understand concepts clearly based strictly on the provided lecture transcript material.

    [CONSTRAINTS]
    1. Grounding: Answer the student's question using the provided context below. If the exact topic isn't found using the exact keywords, look for closely related concepts or overview topics mentioned in the text and explain them.
    2. Fallback: Only if the context is completely unrelated to the question, explicitly say: "I couldn't find this topic in the uploaded lecture material."
    3. Tone: Academic, encouraging, and precise.

    [OUTPUT FORMAT]
    - **Direct Answer:** Clear and structured explanation.
    - **Reference Timestamp(s):** Formatted MM:SS timestamp range where this is taught.

    ---
    [LECTURE CONTEXT]
    {context}

    [STUDENT QUESTION]
    {query}

    [YOUR TUTOR RESPONSE]:
    """
    
    print("[INFO] Groq LLM se response liya ja raha hai...")
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content