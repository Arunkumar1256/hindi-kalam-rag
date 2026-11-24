from __future__ import annotations
import json, os, re, uuid, shutil, threading
from pathlib import Path
from datetime import datetime
from tempfile import NamedTemporaryFile

import chromadb
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from sentence_transformers import SentenceTransformer

from src.rag import ROOT, DB_DIR, MODEL_NAME, extract_pages, make_chunks, normalize, HindiRAG

APP_DIR = ROOT / "document_workspace"
PDF_DIR = APP_DIR / "pdfs"
MANIFEST = APP_DIR / "documents.json"
APP_DIR.mkdir(exist_ok=True)
PDF_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)

LOCK = threading.Lock()
app = FastAPI(title="Document AI Workspace")
app.mount("/static", StaticFiles(directory=str(ROOT / "web_static")), name="static")
client = chromadb.PersistentClient(path=str(DB_DIR))
embedder = SentenceTransformer(MODEL_NAME)


def read_manifest():
    if not MANIFEST.exists(): return []
    try: return json.loads(MANIFEST.read_text(encoding="utf-8"))
    except Exception: return []

def save_manifest(docs):
    MANIFEST.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")

def safe_filename(name):
    base = Path(name or "document.pdf").name
    return re.sub(r"[^A-Za-z0-9._ -]", "_", base)[:120] or "document.pdf"

@app.get("/")
def home():
    return FileResponse(ROOT / "web_static" / "index.html")


@app.get("/api/documents")
def documents():
    out=[]
    for d in read_manifest():
        try: count=client.get_collection(d["collection"]).count()
        except Exception: count=0
        out.append({**d, "chunks": count})
    return {"documents":out}


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    filename=safe_filename(file.filename)
    if not filename.lower().endswith(".pdf"): raise HTTPException(400,"Please upload a PDF file.")
    contents=await file.read()
    if not contents: raise HTTPException(400,"The uploaded PDF is empty.")
    if len(contents)>50*1024*1024: raise HTTPException(413,"PDF is too large (max 50 MB).")
    doc_id=uuid.uuid4().hex
    collection_name=f"pdf_{doc_id[:24]}"
    stored=PDF_DIR / f"{doc_id}.pdf"
    stored.write_bytes(contents)
    try:
        pages=extract_pages(stored)
        chunks=make_chunks(pages)
        if not chunks: raise ValueError("No selectable text found. Scanned PDFs need OCR first.")
        vectors=await run_in_threadpool(lambda: embedder.encode([f"passage: {c.text}" for c in chunks], normalize_embeddings=True, show_progress_bar=True).tolist())
        collection=client.create_collection(name=collection_name, metadata={"hnsw:space":"cosine"})
        collection.upsert(ids=[c.chunk_id for c in chunks], documents=[c.text for c in chunks], embeddings=vectors,
            metadatas=[{"page":c.page,"section":c.section,"chunk_id":c.chunk_id} for c in chunks])
        item={"id":doc_id,"filename":filename,"collection":collection_name,"stored_file":stored.name,
              "uploaded_at":datetime.now().isoformat(timespec="seconds"),"pages":len(pages)}
        with LOCK:
            docs=read_manifest(); docs.append(item); save_manifest(docs)
        return {"document":{**item,"chunks":len(chunks)},"message":"PDF processed and indexed successfully."}
    except Exception as e:
        try: client.delete_collection(collection_name)
        except Exception: pass
        stored.unlink(missing_ok=True)
        raise HTTPException(422,str(e))

@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id:str):
    with LOCK:
        docs=read_manifest(); found=next((d for d in docs if d["id"]==doc_id),None)
        if not found: raise HTTPException(404,"Document not found.")
        try: client.delete_collection(found["collection"])
        except Exception: pass
        (PDF_DIR / found["stored_file"]).unlink(missing_ok=True)
        save_manifest([d for d in docs if d["id"] != doc_id])
    return {"deleted":True}

class Question(BaseModel):
    question: str


@app.post("/api/documents/{doc_id}/ask")
async def ask_document(doc_id:str, body:Question):
    query=normalize(body.question)
    if not query: raise HTTPException(400,"Enter a question.")
    doc=next((d for d in read_manifest() if d["id"]==doc_id),None)
    if not doc: raise HTTPException(404,"Document not found.")
    collection=client.get_collection(doc["collection"])
    if collection.count()==0: raise HTTPException(409,"This PDF has not been indexed.")
    vector=await run_in_threadpool(lambda: embedder.encode([f"query: {query}"], normalize_embeddings=True)[0].tolist())
    result=collection.query(query_embeddings=[vector], n_results=min(4,collection.count()), include=["documents","metadatas","distances"])
    hits=[]
    for text,meta,distance in zip(result["documents"][0],result["metadatas"][0],result["distances"][0]):
        hits.append({"text":text,"metadata":meta,"score":round(1-float(distance),4)})
    answer=await run_in_threadpool(HindiRAG.answer, query, hits)
    sources=[{"page":h["metadata"]["page"],"section":h["metadata"]["section"],"score":h["score"]} for h in hits]
    return {"answer":answer,"sources":sources}
