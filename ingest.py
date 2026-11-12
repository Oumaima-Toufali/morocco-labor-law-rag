import glob, re, chromadb
from pypdf import PdfReader
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
client = chromadb.PersistentClient("db")

# Repartir d'une base propre à chaque ingestion
try:
    client.delete_collection("docs")
except Exception:
    pass
col = client.create_collection("docs", embedding_function=ef)

def clean(text):
    text = re.sub(r"\n-\s*\d+\s*-\n", "\n", text)   # numéros de page "-74 -"
    text = re.sub(r"[ \t]+", " ", text)
    return text

def chunk(text, max_size=2000, overlap=150):
    parts = re.split(r"(?=\bArticle\s+\d+)", text)
    chunks = []
    for p in parts:
        p = p.strip()
        if not p.startswith("Article"):      # supprime préface, sommaire, etc.
            continue
        while len(p) > max_size:
            chunks.append(p[:max_size])
            p = p[max_size - overlap:]
        if p:
            chunks.append(p)
    return chunks

for f in glob.glob("data/*.pdf"):
    text = clean("\n".join(p.extract_text() or "" for p in PdfReader(f).pages))
    chunks = chunk(text)
    col.add(
        documents=chunks,
        ids=[f"{f}-{i}" for i in range(len(chunks))],
        metadatas=[{"source": f}] * len(chunks),
    )
print("Done:", col.count(), "chunks")