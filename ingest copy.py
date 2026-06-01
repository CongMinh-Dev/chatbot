# -*- coding: utf-8 -*-
import os
import glob
import shutil
import asyncio
from tqdm import tqdm
from lightrag import LightRAG
from lightrag.utils import wrap_embedding_func_with_attrs
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

WORKING_DIR = "./lightrag_db"

@wrap_embedding_func_with_attrs(
    embedding_dim=768,      # Gemma embedding dùng 768 chiều
    max_token_size=8192,
    model_name="embeddinggemma"
)
# Hàm bọc để chỉ định model embedding cho Ollama
async def custom_ollama_embed(texts, **kwargs):
    return await ollama_embed(texts, model="embeddinggemma", **kwargs)

async def main():
    if os.path.exists(WORKING_DIR):
        print("🧹 Đang xóa sạch dữ liệu cũ...")
        shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR, exist_ok=True)

    print("🧠 Đang khởi tạo LightRAG kết nối Ollama (Local)...")
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="gemma4-local", # Tên model LLM của bạn trong Ollama
        embedding_func=custom_ollama_embed, 
    )
    
    rag.chunk_size = 500
    rag.chunk_overlap = 100
    await rag.initialize_storages()

    print("📂 Quét tài liệu...")
    docs_text = []
    file_paths = glob.glob("./papers/**/*.*", recursive=True)

    for path in file_paths:
        try:
            ext = os.path.splitext(path)[-1].lower()
            if ext == ".pdf":
                pages = PyPDFLoader(path).load()
                docs_text.append("\n".join([p.page_content for p in pages]))
            elif ext in [".txt", ".md"]:
                docs_text.append(TextLoader(path, encoding="utf-8").load()[0].page_content)
        except Exception as e:
            print(f"⚠️ Lỗi đọc file {path}: {e}")

    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_text("\n\n".join(docs_text))
    
    print(f"⚡ Bắt đầu xây dựng đồ thị với {len(chunks)} phân đoạn...")
    with tqdm(total=len(chunks), desc="🤖 Đang xử lý") as pbar:
        for chunk in chunks:
            await rag.ainsert(chunk)
            pbar.update(1)
            
    print("\n✅ Hoàn tất nạp dữ liệu!")

if __name__ == "__main__":
    asyncio.run(main())