"""
Document Assistant - Flask app (RAG with a local Ollama model)

Setup:
    pip install flask pypdf python-docx sentence-transformers chromadb ollama
    ollama pull qwen2.5:7b

Run:
    python app.py
Then open http://127.0.0.1:5000
"""

import os
import re
import tempfile

import chromadb
import ollama
from docx import Document
from flask import Flask, jsonify, render_template, request
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024  

LLM_MODEL = "qwen2.5:7b"
ALLOWED = (".pdf", ".docx", ".txt")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma = chromadb.Client()  
state = {"collection": None}


# ---------- Reading and chunking ----------
def extract_pages(path):
    lower = path.lower()
    if lower.endswith(".pdf"):
        reader = PdfReader(path)
        return [
            {"page": i + 1, "text": page.extract_text() or ""}
            for i, page in enumerate(reader.pages)
        ]
    if lower.endswith(".docx"):
        doc = Document(path)
        return [{"page": 1, "text": "\n".join(p.text for p in doc.paragraphs)}]
    with open(path, encoding="utf-8", errors="ignore") as f:
        return [{"page": 1, "text": f.read()}]


def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text, size=800, overlap=150):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(". ", start, end)
            if cut > start + size // 2:
                end = cut + 1
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_chunks(path):
    result = []
    for p in extract_pages(path):
        for c in chunk_text(clean_text(p["text"])):
            result.append({"page": p["page"], "text": c})
    return result


# ---------- Routes ----------
@app.get("/")
def index():
    return render_template("index.html")


@app.post("/upload")
def upload():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify(error="Choose a file first."), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED:
        return jsonify(error="Use a PDF, Word (.docx) or text file."), 400

    fd, tmp_path = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    try:
        file.save(tmp_path)
        chunks = build_chunks(tmp_path)
    except Exception:
        return jsonify(error="This file could not be read."), 422
    finally:
        os.remove(tmp_path)

    if not chunks:
        return jsonify(
            error="No text found."
        ), 422

    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts).tolist()
    try:
        chroma.delete_collection("docs")
    except Exception:
        pass
    collection = chroma.create_collection("docs")

    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"page": c["page"]} for c in chunks],
    )
    state["collection"] = collection

    return jsonify(filename=file.filename, chunks=len(chunks), pages = len(set(c['page'] for c in chunks)))

INTENT_PROMPT = """
You are an intent classifier for a book assistant.

Classify the user's input into exactly one of these categories:

1. QUESTION: the user asks a question about the uploaded book.
2. SUMMARY: the user wants to summarize the book, a chapter, or a specific section.
3. UNKNOWN: the input does not match either category, is unclear, or is unrelated to the book.

Examples:
"What is a kernel in SVM?" -> QUESTION
"What is chapter 3 about?" -> QUESTION
"Summarize chapter 3" -> SUMMARY
"Give me an overview of the section about trees" -> SUMMARY
"What's the weather today?" -> UNKNOWN

Return ONLY one word, with no explanation: QUESTION, SUMMARY, or UNKNOWN.

User input:
{user_input}
"""


def llm(prompt, temperature=0.2, num_ctx=4096):
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature, "num_ctx": num_ctx},
    )
    return response["message"]["content"]


SUMMARY_WORDS = ("summar", "overview", "tl;dr", "tldr", "recap",
                 "main points", "key points")
SMALL_TALK = ("hi", "hello", "hey", "thanks", "thank you")

def classify_intent(user_input):
    t = user_input.lower().strip()
    if any(w in t for w in SUMMARY_WORDS):
        return "SUMMARY"
    if t.strip(" !.?") in SMALL_TALK:
        return "UNKNOWN"
    return "QUESTION"


def retrieve(collection, text, n):
    """أقرب n قطعة، مرتبة بترتيبها الأصلي في الكتاب."""
    res = collection.query(
        query_embeddings=embedder.encode([text]).tolist(),
        n_results=min(n, collection.count()),
    )
    items = zip(res["ids"][0], res["documents"][0], res["metadatas"][0])
    return sorted(items, key=lambda item: int(item[0]))


def build_context(items):
    return "\n\n".join(f"[Page {m['page']}]\n{d}" for _, d, m in items)


def answer_question(collection, question):
    items = retrieve(collection, question, 3)
    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I couldn't find this in the document."
Be concise: 2-4 sentences.

Context:
{build_context(items)}

Question: {question}"""
    pages = sorted({m["page"] for _, _, m in items})
    return llm(prompt), pages


def summarize(collection, request_text):
    items = retrieve(collection, request_text, 8)
    prompt = f"""Summarize the text below, following this request: {request_text}
Use ONLY the context. Write a clear summary of 4-6 sentences.

Context:
{build_context(items)}"""
    pages = sorted({m["page"] for _, _, m in items})
    return llm(prompt, num_ctx=8192), pages


@app.post("/ask")
def ask():
    collection = state["collection"]
    if collection is None:
        return jsonify(error="Upload a document first."), 400

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if len(question) < 3:
        return jsonify(error="Type a question first."), 400

    try:
        intent = classify_intent(question)
        if intent == "SUMMARY":
            answer, pages = summarize(collection, question)
        elif intent == "QUESTION":
            answer, pages = answer_question(collection, question)
        else:
            answer = "I can answer questions about the document or summarize parts of it. Try rephrasing your request."
            pages = []
    except Exception:
        return jsonify(
            error="Could not reach Ollama. Make sure it is running and the model is downloaded."
        ), 502

    return jsonify(answer=answer, pages=pages)



if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, use_reloader=False)
