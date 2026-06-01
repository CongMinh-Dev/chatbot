# -*- coding: utf-8 -*-
import os
import shutil
import asyncio
from pathlib import Path
from tqdm import tqdm
from lightrag import LightRAG
from lightrag.utils import wrap_embedding_func_with_attrs
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 🛠️ SỬA ĐƯỜNG DẪN THƯ MỤC CƠ SỞ DỮ LIỆU ĐỒ THỊ
WORKING_DIR = "/content/chatbot/lightrag_db"

@wrap_embedding_func_with_attrs(
    embedding_dim=768,
    max_token_size=8192,
    model_name="embeddinggemma"
)
async def custom_ollama_embed(texts, **kwargs):
    return await ollama_embed(texts, model="embeddinggemma", **kwargs)

async def main():
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR, exist_ok=True)

    print("🧠 Đang khởi tạo LightRAG kết nối Ollama...")
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="qwen3.5:9b", 
        embedding_func=custom_ollama_embed, 
        addon_params={
            "language": "Vietnamese",
            "entity_relationship_graph_type": "default"
        }
    )
    
    rag.chunk_size = 500
    rag.chunk_overlap = 100
    rag.max_gleaning = 0
    await rag.initialize_storages()

    print("📂 Quét tài liệu...")
    docs_text = []
    
    # 🛠️ SỬA ĐƯỜNG DẪN THƯ MỤC CHỨA TÀI LIỆU CỦA BẠN
    source_dir = Path("/content/chatbot/papers")
    
    if not source_dir.exists():
        print(f"❌ Không tìm thấy thư mục tại đường dẫn: {source_dir}")
        return
        
    # Tìm kiếm đệ quy toàn bộ file tài liệu
    file_paths = []
    for ext in ["**/*.pdf", "**/*.txt", "**/*.md"]:
        file_paths.extend(list(source_dir.glob(ext)))
        
    print(f"🔍 Tìm thấy tổng cộng {len(file_paths)} file tài liệu tại {source_dir}")

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
                else:
                    print(f"⚠️ Cảnh báo: File PDF {path_obj.name} bị rỗng (hoặc file quét ảnh dạng scan)")
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

    # Tiến hành chia nhỏ văn bản
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text("\n\n".join(docs_text))
    
    print(f"⚡ Bắt đầu xây dựng đồ thị với {len(chunks)} phân đoạn...")
    
    for i, chunk in enumerate(tqdm(chunks, desc="🤖 Đang xử lý")):
        try:
            await rag.ainsert(chunk)
        except Exception as e:
            print(f"\n❌ Lỗi tại chunk {i+1}: {e}")
            
    print("\n✅ Hoàn tất nạp dữ liệu!")
    
    # Đóng gói file zip và lưu ngay trong thư mục chatbot cho bạn dễ quản lý
    zip_output_path = "/content/chatbot/lightrag_db_exported"
    shutil.make_archive(zip_output_path, 'zip', WORKING_DIR)
    print(f"📦 Đã đóng gói thành công file đồ thị tại: {zip_output_path}.zip")

if __name__ == '__main__':
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())
    except RuntimeError:
        asyncio.run(main())