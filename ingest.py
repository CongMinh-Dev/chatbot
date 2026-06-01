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

WORKING_DIR = "/content/chatbot/lightrag_db"
ZIP_OUTPUT_PATH = "/content/chatbot/lightrag_db_exported"

@wrap_embedding_func_with_attrs(
    embedding_dim=768,
    max_token_size=8192,
    model_name="embeddinggemma"
)
async def custom_ollama_embed(texts, **kwargs):
    return await ollama_embed(texts, model="embeddinggemma", **kwargs)

async def main():
    # ❌ KHÔNG XÓA THƯ MỤC CŨ NỮA ĐỂ GIỮ CHECKPOINT KHHI COLAB BỊ TẮT
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR, exist_ok=True)
        print("📁 Đã tạo thư mục lưu trữ đồ thị mới.")
    else:
        print("🔄 Phát hiện dữ liệu cũ! Chế độ RESUME (Chạy tiếp tục) đã được kích hoạt.")

    print("🧠 Đang khởi tạo LightRAG kết nối Ollama...")
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="qwen3.5:9b", 
        embedding_func=custom_ollama_embed, 
        
        # Giữ số worker an toàn cho GPU T4 local
        llm_async_max_workers=1,
        embedding_async_max_workers=8,
        
        addon_params={
            "language": "Vietnamese",
            "entity_relationship_graph_type": "default"
        }
    )
    
    # Giữ nguyên độ chính xác cao mặc định của bạn (không dùng tinh giản)
    # RAG sẽ quét kỹ để trích xuất sâu các mối quan hệ râu ria
    await rag.initialize_storages()

    print("📂 Quét tài liệu tại thư mục /content/chatbot/papers ...")
    docs_text = []
    source_dir = Path("/content/chatbot/papers")
    
    if not source_dir.exists():
        print(f"❌ Không tìm thấy thư mục tại đường dẫn: {source_dir}")
        return
        
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
                    print(f"📖 Đã nạp PDF: {path_obj.name}")
            elif ext in [".txt", ".md"]:
                content = TextLoader(path, encoding="utf-8").load()[0].page_content
                if content.strip():
                    docs_text.append(content)
                    print(f"📖 Đã nạp Text: {path_obj.name}")
        except Exception as e:
            print(f"⚠️ Lỗi đọc file {path_obj.name}: {e}")

    if not docs_text:
        print("❌ Không có dữ liệu để phân mảnh!")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text("\n\n".join(docs_text))
    
    print(f"⚡ Tổng số phân đoạn cần kiểm tra/xử lý: {len(chunks)}")
    
    # Vòng lặp xử lý từng chunk và auto-save liên tục
    for i, chunk in enumerate(tqdm(chunks, desc="🤖 Đang xử lý")):
        try:
            # Hàm ainsert sẽ tự check md5/chuỗi hash của chunk, nếu trùng trong DB nó sẽ bỏ qua rất nhanh
            await rag.ainsert(chunk)
            
            # 📦 CỨ XỬ LÝ XONG 1 CHUNK LÀ NÉN LẠI NGAY LẬP TỨC
            shutil.make_archive(ZIP_OUTPUT_PATH, 'zip', WORKING_DIR)
            
        except Exception as e:
            print(f"\n❌ Lỗi tại chunk {i+1}: {e}")
            
    print("\n✅ HOÀN TẤT TOÀN BỘ TIẾN TRÌNH!")
    print(f"📦 File đồ thị cuối cùng đã sẵn sàng tại: {ZIP_OUTPUT_PATH}.zip")

if __name__ == '__main__':
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())
    except RuntimeError:
        asyncio.run(main())