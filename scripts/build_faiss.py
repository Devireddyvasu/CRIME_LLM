import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# ==============================
# LOAD JSONL
# ==============================
docs = []

with open("data/processed/crime_processed_data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        docs.append(Document(page_content=data["text"]))

print(f"Loaded {len(docs)} documents")

# ==============================
# CHUNKING
# ==============================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

print(f"Created {len(chunks)} chunks")
for i in range(min(4, len(chunks))):
    print(f"\nChunk {i+1}:\n")
    print(chunks[i].page_content)
    print("------")
# ==============================
# EMBEDDING MODEL
# ==============================
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# ==============================
# SHOW EMBEDDING OUTPUT (IMPORTANT)
# ==============================

sample_text = chunks[0].page_content

vector = embedding.embed_query(sample_text)

print("\nEmbedding vector (first 10 values):\n")
print(vector[:10])

print("\nEmbedding dimension:", len(vector))
# # ==============================
# # BUILD FAISS
# # ==============================

db = FAISS.from_documents(chunks, embedding)

# # ==============================
# # SAVE DB
# # ==============================
db.save_local("vector_db")
print("\nNumber of vectors stored in FAISS:")
print(db.index.ntotal)
print("✅ FAISS index built successfully!")

# ==============================
# RETRIEVAL (Step 5)
# ==============================
query = "Which area has the most theft cases?"

results = db.similarity_search(query, k=3)

print("\nRetrieved Documents:\n")

for i, doc in enumerate(results, 1):
    print(f"Result {i}:\n")
    print(doc.page_content)
    print("------")