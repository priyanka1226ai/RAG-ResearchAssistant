# ingest.py
import os
import uuid
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm

# ---------------- Config ----------------
DATA_DIR = "data"                  # Folder containing PDFs/TXT files
PERSIST_DIR = "chroma_db"          # Chroma persistence directory
MODEL_NAME = "all-MiniLM-L6-v2"    # Embedding model
CHUNK_SIZE = 1000                  # Characters per chunk
CHUNK_OVERLAP = 200                # Overlap between chunks

# ---------------- Helpers ----------------
def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end - overlap, end)  # move window with overlap
    return chunks


def extract_text_from_pdf(path):
    """Extract all text from a PDF file"""
    reader = PdfReader(path)
    out = []
    for p in reader.pages:
        txt = p.extract_text() or ""
        out.append(txt)
    return "\n".join(out)


# ---------------- Setup embedding model & Chroma ----------------
print("Loading embedding model...")
embedder = SentenceTransformer(MODEL_NAME)

print("Starting Chroma client...")
client = chromadb.PersistentClient(path=PERSIST_DIR)

collection_name = "research_docs"
try:
    collection = client.get_collection(collection_name)
except Exception:
    collection = client.create_collection(collection_name)

# ---------------- Walk files and ingest ----------------
all_files = []
for root, _, files in os.walk(DATA_DIR):
    for f in files:
        if f.lower().endswith(".pdf") or f.lower().endswith(".txt"):
            all_files.append(os.path.join(root, f))

ids, documents, metadatas = [], [], []

for path in tqdm(all_files, desc="Files"):
    print("Processing", path)
    if path.lower().endswith(".pdf"):
        text = extract_text_from_pdf(path)
    else:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()

    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    for i, chunk in enumerate(chunks):
        uid = str(uuid.uuid4())
        ids.append(uid)
        documents.append(chunk)
        metadatas.append({"source": os.path.relpath(path), "chunk": i})

# ---------------- Embedding & Adding ----------------
print("Embedding ...")
batch_size = 64
embeddings = []
for i in tqdm(range(0, len(documents), batch_size)):
    batch = documents[i: i + batch_size]
    embs = embedder.encode(batch, convert_to_numpy=True)
    embeddings.extend(embs.tolist())

print("Adding to Chroma collection ...")
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids,
    embeddings=embeddings,
)

print("✅ Done. Ingested:", len(documents), "chunks into collection:", collection_name)
