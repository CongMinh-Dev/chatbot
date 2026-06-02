# -*- coding: utf-8 -*-
import os
import shutil
import asyncio
from pathlib import Path
from tqdm import tqdm

# ========================================================
# 🔥 ĐOẠN VÁ LỖI TIKTOKEN CHẶN <|endoftext|> TRÊN COLAB
# ========================================================
import tiktoken
_orig_encode = tiktoken.Encoding.encode
def safe_encode(self, text, *args, **kwargs):
    if 'allowed_special' not in kwargs and 'disallowed_special' not in kwargs:
        kwargs['allowed_special'] = 'all'
        kwargs['disallowed_special'] = ()
    return _orig_encode(self, text, *args, **kwargs)
tiktoken.Encoding.encode = safe_encode
# ========================================================

from lightrag import LightRAG
from lightrag.utils import wrap_embedding_func_with_attrs
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Cấu hình đường dẫn tuyệt đối theo đúng cấu trúc thư mục của bạn
WORKING_DIR = "/content/chatbot/lightrag_db"

@wrap_embedding_func_with_attrs(
    embedding_dim=768,
    max_token_size=8192,
    model_name="embeddinggemma"
)
async def custom_ollama_embed(texts, **kwargs):
    return await ollama_embed(texts, model="embeddinggemma", **kwargs)

async def main():
    # Xóa dữ liệu lỗi cũ để khởi động lại một cách sạch sẽ
    if os.path.exists(WORKING_DIR):
        print("🧹 Đang làm sạch thư mục đồ thị cũ...")
        shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR, exist_ok=True)

    print("🧠 Đang khởi tạo LightRAG kết nối Ollama...")
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="qwen2.5:7b", 
        embedding_func=custom_ollama_embed, 
        addon_params={
            "language": "Vietnamese",
            "entity_relationship_graph_type": "default"
        }
    )
    rag.chunk_size = 200
    rag.chunk_overlap = 100
    rag.max_gleaning = 0
    await rag.initialize_storages()

    print("📂 Quét tài liệu tại thư mục /content/chatbot/papers ...")
    docs_text = []
    source_dir = Path("/content/chatbot/papers")
    
    if not source_dir.exists():
        print(f"❌ Không tìm thấy thư mục tại đường dẫn: {source_dir}")
        return
        
    # Tìm kiếm đệ quy toàn bộ file tài liệu
    file_paths = []
    for ext in ["**/*.pdf", "**/*.txt", "**/*.md"]:
        file_paths.extend(list(source_dir.glob(ext)))
        
    print(f"🔍 Tìm thấy tổng cộng {len(file_paths)} file tài liệu.")

    for path_obj in file_paths:
        path = str(path_obj)
        try:
            ext = path_obj.suffix.lower()
            if ext == ".pdf":
                pages = PyPDFLoader(path).load()
                content = "\n".join([p.page_content for p in pages])
                if content.strip():
                    docs_text.append(content)
                    print(f"📖 Đã nạp thành công PDF: {path_obj.name}")
            elif ext in [".txt", ".md"]:
                content = TextLoader(path, encoding="utf-8").load()[0].page_content
                if content.strip():
                    docs_text.append(content)
                    print(f"📖 Đã nạp thành công Text: {path_obj.name}")
        except Exception as e:
            print(f"⚠️ Lỗi khi cố đọc file {path_obj.name}: {e}")

    if not docs_text:
        print("❌ Không có bất kỳ dữ liệu văn bản nào được trích xuất thành công để phân mảnh!")
        return

    # Tiến hành chia nhỏ văn bản thành các chunks 500 ký tự
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text("\n\n".join(docs_text))
    
    print(f"⚡ Bắt đầu xây dựng đồ thị với {len(chunks)} phân đoạn...")
    
    # Nạp trực tiếp từng phân đoạn vào LightRAG thông qua cơ chế bất đồng bộ
    zip_output_path = "/content/chatbot/lightrag_snapshots"
    for i, chunk in enumerate(tqdm(chunks, desc="🤖 Đang xử lý")):
        try:
            await rag.ainsert(chunk)
            await asyncio.sleep(2)
            dynamic_zip_path = f"{zip_output_path}/lightrag_chunk_{i+1}"
            shutil.make_archive(dynamic_zip_path, 'zip', WORKING_DIR)
            print(f"📦 Đã đóng gói thành công file đồ thị tại: {zip_output_path}.zip")
        except Exception as e:
            print(f"\n❌ Lỗi tại chunk {i+1}: {e}")
            
    print("\n✅ Hoàn tất nạp dữ liệu!")
    
    

if __name__ == '__main__':
    try:
        # Colab chạy sẵn một event loop nền, cần kiểm tra để tránh xung đột
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())
    except RuntimeError:
        asyncio.run(main())