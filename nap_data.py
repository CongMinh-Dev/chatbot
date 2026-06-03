# -*- coding: utf-8 -*-
import os
import shutil
import asyncio
import logging  # <-- Thư viện để quản lý và bắt log bằng chữ
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
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

WORKING_DIR = "/content/chatbot/lightrag_db"
checkpoint_file = os.path.join(WORKING_DIR, "resume_checkpoint.txt")

# Khai báo các biến toàn cục để bộ bắt log có thể dùng chung với hàm main
CURRENT_CHUNK_INDEX = 0
LOGGED_PERSIST_SUCCESS = False

# ========================================================
# 🎯 BỘ ĐỌC LOG VÀ SO SÁNH CHỮ TỰ ĐỘNG
# ========================================================
class LightRAGLogInterceptor(logging.Handler):
    def emit(self, record):
        global LOGGED_PERSIST_SUCCESS
        
        # 1. Lấy ra nội dung thông báo bằng chữ (String) từ hệ thống log
        log_message = record.getMessage()
        
        # 2. So sánh trực tiếp bằng chữ xem có đúng cụm từ yêu cầu không
        if "In memory DB persist to disk" in log_message:
            # Đánh dấu là đã tìm thấy dòng chữ thành công
            LOGGED_PERSIST_SUCCESS = True
            
            # 3. Tiến hành ghi luôn index của chunk xuống file txt
            try:
                with open(checkpoint_file, "w") as f:
                    # CURRENT_CHUNK_INDEX + 1 nghĩa là lần sau mở lên sẽ chạy chunk tiếp theo
                    f.write(str(CURRENT_CHUNK_INDEX + 1))
                print(f"\n💾 [Hệ thống Log] Phát hiện chữ 'In memory DB persist to disk' -> Đã chốt sổ chunk {CURRENT_CHUNK_INDEX + 1} vào txt.")
            except Exception as e:
                print(f"\n❌ Lỗi khi ghi file checkpoint từ Log: {e}")

# Kích hoạt bộ lọc log này để nghe trọn vẹn thư viện lightrag
logger = logging.getLogger("lightrag")
logger.addHandler(LightRAGLogInterceptor())
# ========================================================



async def main():
    global CURRENT_CHUNK_INDEX, LOGGED_PERSIST_SUCCESS
    
    os.makedirs(WORKING_DIR, exist_ok=True)

    print("🧠 Đang khởi tạo LightRAG kết nối Ollama...")
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="qwen2.5:14b",
        embedding_func=ollama_embed, 
        addon_params={
            "language": "Vietnamese",
            "entity_relationship_graph_type": "default"
        }
    )
    rag.chunk_size = 500
    rag.chunk_overlap = 100
    rag.max_gleaning = 0
    await rag.initialize_storages()

    # Đọc checkpoint cũ để bỏ qua các chunk đã xong lần trước
    start_chunk_index = 0
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                start_chunk_index = int(f.read().strip())
            print(f"🔄 Đang tiếp tục tiến trình cũ! Sẽ bắt đầu chạy từ chunk thứ: {start_chunk_index + 1}")
        except Exception:
            pass

    print("📂 Quét tài liệu tại thư mục /content/chatbot/papers ...")
    docs_text = []
    source_dir = Path("/content/chatbot/papers")
    
    if not source_dir.exists():
        print(f"❌ Không tìm thấy thư mục tại đường dẫn: {source_dir}")
        return
        
    file_paths = []
    for ext in ["**/*.pdf", "**/*.txt", "**/*.md"]:
        file_paths.extend(list(source_dir.glob(ext)))
        
    for path_obj in file_paths:
        path = str(path_obj)
        try:
            ext = path_obj.suffix.lower()
            if ext == ".pdf":
                pages = PyPDFLoader(path).load()
                content = "\n".join([p.page_content for p in pages])
                if content.strip(): docs_text.append(content)
            elif ext in [".txt", ".md"]:
                content = TextLoader(path, encoding="utf-8").load()[0].page_content
                if content.strip(): docs_text.append(content)
        except Exception as e:
            print(f"⚠️ Lỗi khi đọc file {path_obj.name}: {e}")

    if not docs_text:
        print("❌ Không có dữ liệu văn bản!")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text("\n\n".join(docs_text))
    total_chunks = len(chunks)

    if start_chunk_index >= total_chunks:
        print("🎉 Toàn bộ dữ liệu đã được xử lý xong từ trước rồi!")
        return
        
    # --- VÒNG LẶP XỬ LÝ CHUNKS ---
    for i in range(start_chunk_index, total_chunks):
        chunk = chunks[i]
        
        # Đồng bộ biến i hiện tại sang biến toàn cục để Bộ đọc log nhận diện đúng số thứ tự chunk
        CURRENT_CHUNK_INDEX = i  
        LOGGED_PERSIST_SUCCESS = False # Reset lại cờ xác nhận trước khi nạp chunk mới
        
        print(f"\n🤖 Đang xử lý chunk {i+1}/{total_chunks}...")
        try:
            await rag.ainsert(chunk)
            
            # Sau khi chạy xong hàm ainsert, kiểm tra xem bộ đọc log có kích hoạt cờ thành công lên không
            if not LOGGED_PERSIST_SUCCESS:
                print(f"⚠️ [Cảnh báo] Chunk {i+1} đã kết thúc lệnh nhưng Bộ đọc log KHÔNG tìm thấy chữ 'In memory DB persist to disk'.")
                print("🛑 Hệ thống tự động dừng để bảo vệ checkpoint!")
                return # Dừng chương trình ngay, giữ nguyên checkpoint cũ
                
            await asyncio.sleep(0.5) 
        except Exception as e:
            print(f"❌ Lỗi crash tại chunk {i+1}: {e}")
            return
            
    # Nếu đã chạy đến đây tức là hoàn thành 100% không lỗi, xóa file checkpoint đi
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    zip_output_path = "/content/chatbot/lightrag_snapshots/lightrag_final_backup"
    os.makedirs("/content/chatbot/lightrag_snapshots", exist_ok=True)
    print("\n📦 Đang đóng gói toàn bộ đồ thị tri thức...")
    shutil.make_archive(zip_output_path, 'zip', WORKING_DIR)
    print(f"✅ Hoàn tất nạp dữ liệu! File của bạn: {zip_output_path}.zip")

if __name__ == '__main__':
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())
    except RuntimeError:
        asyncio.run(main())