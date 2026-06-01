# -*- coding: utf-8 -*-
import os
import shutil
import asyncio
from pathlib import Path
from tqdm import tqdm
from datetime import datetime  # Thêm thư viện để lấy thời gian thực cho tên file

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

# ========================================================
# 📁 CẤU HÌNH ĐƯỜNG DẪN (Nên đổi sang Google Drive nếu cần)
# ========================================================
WORKING_DIR = "/content/chatbot/lightrag_db"
CHECKPOINT_FILE = os.path.join(WORKING_DIR, "ingest_checkpoint.txt")

# Nơi chứa các file zip backup (Sẽ tạo thành một thư mục riêng để dễ quản lý)
ZIP_BACKUP_DIR = "/content/chatbot/lightrag_snapshots" 
SAVE_EVERY_N_CHUNKS = 1  # Cứ sau 1 chunks sẽ nén một file mới

@wrap_embedding_func_with_attrs(
    embedding_dim=768,
    max_token_size=8192,
    model_name="embeddinggemma"
)
async def custom_ollama_embed(texts, **kwargs):
    return await ollama_embed(texts, model="embeddinggemma", **kwargs)

def save_progress_and_zip(current_index):
    """Ghi nhận vị trí đã xử lý và tiến hành đóng gói dữ liệu THÀNH FILE MỚI (Không ghi đè)"""
    # 1. Ghi lại index tiếp theo cần xử lý vào file checkpoint cục bộ
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        f.write(str(current_index))
    
    # 2. Tạo tên file zip độc nhất dựa trên số chunk và thời gian thực
    timestamp = datetime.now().strftime("%Hh%Mm%Ss")
    zip_filename = f"lightrag_db_chunk_{current_index}_{timestamp}"
    full_zip_path = os.path.join(ZIP_BACKUP_DIR, zip_filename)
    
    # 3. Tiến hành nén liên tiếp
    shutil.make_archive(full_zip_path, 'zip', WORKING_DIR)
    print(f"\n💾 Đã lưu checkpoint tại chunk {current_index} | Cập nhật file backup mới: {full_zip_path}.zip")

async def main():
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(ZIP_BACKUP_DIR, exist_ok=True)

    # Đọc checkpoint cũ nếu có
    start_chunk_index = 0
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                start_chunk_index = int(f.read().strip())
            print(f"🔄 Tìm thấy tiến trình cũ. Sẽ tiếp tục chạy từ chunk thứ: {start_chunk_index + 1}")
        except Exception:
            print("⚠️ File checkpoint lỗi hoặc không đọc được. Sẽ chạy lại từ đầu.")
            start_chunk_index = 0

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
    
    # Tự động nạp lại các file đồ thị cũ (.json) từ WORKING_DIR nếu có sẵn
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

    # Sắp xếp để đảm bảo thứ tự chunk luôn đồng nhất giữa các lần chạy lại
    file_paths.sort() 
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

    # Tiến hành chia nhỏ văn bản thành các chunks 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text("\n\n".join(docs_text))
    
    total_chunks = len(chunks)
    print(f"⚡ Tổng số phân đoạn tìm thấy: {total_chunks}")
    
    if start_chunk_index >= total_chunks:
        print("✅ Toàn bộ các chunk đã được xử lý xong từ trước đó!")
        return

    # Chỉ xử lý phần còn lại chưa chạy
    remaining_chunks = chunks[start_chunk_index:]
    print(f"🚀 Bắt đầu xử lý tiếp từ chunk {start_chunk_index + 1}/{total_chunks}...")

    # Vòng lặp nạp dữ liệu
    for i, chunk in enumerate(tqdm(remaining_chunks, desc="🤖 Đang xử lý")):
        actual_index = start_chunk_index + i
        try:
            await rag.ainsert(chunk)
            
            # Kiểm tra định kỳ sau mỗi N chunks
            if (i + 1) % SAVE_EVERY_N_CHUNKS == 0:
                # Ép LightRAG lưu toàn bộ dữ liệu tạm thời từ RAM xuống disk trước khi nén
                await rag.kv_storage.index_done_callback()
                save_progress_and_zip(actual_index + 1)
                
        except Exception as e:
            print(f"\n❌ Lỗi tại chunk {actual_index + 1}: {e}")
            save_progress_and_zip(actual_index)
            
    # Kết thúc xử lý hoàn chỉnh toàn bộ script
    await rag.kv_storage.index_done_callback()
    save_progress_and_zip(total_chunks)
    print("\n✅ Hoàn tất toàn bộ tiến trình nạp dữ liệu!")

if __name__ == '__main__':
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())
    except RuntimeError:
        asyncio.run(main())