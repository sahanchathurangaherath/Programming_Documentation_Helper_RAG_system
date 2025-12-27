# fix_chroma.py
import shutil
import os

# Delete the entire chroma directory
if os.path.exists("data/chroma_db"):
    shutil.rmtree("data/chroma_db")
    print("Deleted old chroma_db")

# Recreate directories
os.makedirs("data/chroma_db", exist_ok=True)
os.makedirs("data/embedding_cache", exist_ok=True)
print("Created fresh directories")

# Now initialize RAG
from rag_system import RAGSystem

rag = RAGSystem()
print("RAG System initialized successfully!")

# Add a test document
rag.add_document("This is a test document about Python programming.", source="test")
print("Added test document")

# Test query
result = rag.query("What is this about?")
print(f"Query result: {result['answer']}")