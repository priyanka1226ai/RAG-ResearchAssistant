# ----------------- Load and validate API key -----------------
import os
from pathlib import Path

# Try environment first
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# If it doesn't start with a service account prefix, load from .env manually
if not OPENAI_API_KEY.startswith("sk-svcacct-"):
    print("❌ Wrong key detected, loading from .env file...")
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                OPENAI_API_KEY = line.split("=", 1)[1].strip()
                break

# Final validation
if not OPENAI_API_KEY.startswith("sk-svcacct-"):
    raise ValueError("❌ Could not find a valid service account OPENAI_API_KEY. "
                     "Make sure your .env contains a key starting with sk-svcacct-")

print(f"DEBUG: Loaded API key starts with: {OPENAI_API_KEY[:12]}")

# Override environment for current session
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# query.py
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI   # ✅ new OpenAI client

# ----------------- Load environment variables -----------------
load_dotenv()  # will load variables from a .env file if present

# ----------------- Configuration -----------------
PERSIST_DIR = "chroma_db"                   # folder where embeddings are stored
EMBED_MODEL = "all-MiniLM-L6-v2"            # embedding model
TOP_K = 5                                   # number of retrieved docs
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ----------------- Initialization -----------------
print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

print("Connecting to Chroma client...")
client = chromadb.PersistentClient(path=PERSIST_DIR)

try:
    collection = client.get_collection("research_docs")
except Exception:
    # if collection not found, create it
    collection = client.create_collection("research_docs")
# ✅ Initialize OpenAI client
raw_key = os.getenv("OPENAI_API_KEY", "")
print("DEBUG: Loaded API key =", repr(raw_key))   # <-- Debug print

if not raw_key:
    raise ValueError("❌ OPENAI_API_KEY not set. Please create a .env file with your API key.")

OPENAI_API_KEY = raw_key.strip()  # ✅ remove whitespace/newlines
openai_client = OpenAI(api_key=OPENAI_API_KEY)




# ----------------- Functions -----------------
def retrieve(query, k=TOP_K):
    """Retrieve top-k most relevant documents from ChromaDB"""
    q_emb = embedder.encode([query], convert_to_numpy=True).tolist()
    results = collection.query(
        query_embeddings=q_emb,
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return list(zip(docs, metas))


def build_prompt(query, retrieved):
    """Builds a prompt for the LLM using retrieved context"""
    prompt_parts = []
    citations = []
    for i, (doc, meta) in enumerate(retrieved, start=1):
        src = meta.get("source", "unknown")
        prompt_parts.append(f"[{i}] Source: {src}\n{doc}\n")
        citations.append(f"[{i}] {src}")

    context = "\n\n".join(prompt_parts)

    instruction = (
        "You are an academic research assistant. Use ONLY the provided sources to answer the user's question. "
        "Cite sources inline using the bracket numbers like [1], [2]. If the answer is not in the documents, say "
        "'I don't know' and propose useful search keywords.\n\n"
    )

    prompt = f"{instruction}Context:\n{context}\n\nUser question: {query}\n\nAnswer (include source citations):"
    return prompt


def ask_openai(prompt, model="gpt-4o-mini", max_tokens=512, temperature=0.0):
    """Send the built prompt to OpenAI GPT model (new API)"""
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful academic research assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content


def answer_query(query):
    """Full pipeline: retrieve → build prompt → ask OpenAI → return answer + sources"""
    retrieved = retrieve(query)
    prompt = build_prompt(query, retrieved)
    ans = ask_openai(prompt)
    sources = [meta["source"] for _, meta in retrieved]
    return ans, sources


# ----------------- CLI Demo -----------------
if __name__ == "__main__":
    q = input("Ask: ")
    ans, sources = answer_query(q)
    print("\n--- Answer ---\n", ans)
    print("\n--- Sources ---")
    for i, src in enumerate(sources, start=1):
        print(f"[{i}] {src}")
