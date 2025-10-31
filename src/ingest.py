import pandas as pd
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

print("Starting ingestion pipeline...")

# 1. Load Data
df = pd.read_csv("/path/aa_dataset-tickets-multi-lang-5-2-50-version.csv")
df.dropna(subset=['answer', 'subject', 'body'], inplace=True)

# 2. Create Documents
documents = []
for _, row in df.iterrows():
    text_to_embed = f"Subject: {row['subject']}\n\nBody: {row['body']}"
    metadata = {
        'answer': row['answer'],
        'language': row['language'],
        'priority': row['priority'],
        'queue': row['queue'],
        'version': str(row['version']), 
        'type': row['type']
    }
    doc = Document(text=text_to_embed, metadata=metadata)
    documents.append(doc)

# 3. Set up ChromaDB (it will save to a folder ./chroma_db)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("ticket_knowledge")

# 4. Set up Embed Model
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# 5. Create and save the index
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

print("Building index... (this may take a moment)")
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model
)

print(f"--- Ingestion complete. Index built and saved to ./chroma_db ---")