from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import os
from dotenv import load_dotenv


query = "sản phẩm hiện có"
load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

embeddings = NVIDIAEmbeddings(
    model="nvidia/nv-embed-v1", # Hoặc model phù hợp khác
    api_key=NVIDIA_API_KEY
    )
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
results = vectorstore.similarity_search(query, k=5)
for i, res in enumerate(results):
    print(f"--- Chunk {i} ---")
    print(res.page_content)