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
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Cấu hình đường dẫn tuyệt đối theo đúng cấu trúc thư mục của bạn
WORKING_DIR = "/content/chatbot/lightrag_db"



# --------------
async def main():
    # ❌ KHÔNG XÓA THƯ MỤC CŨ NỮA! Để dữ liệu cũ được giữ nguyên.
    os.makedirs(WORKING_DIR, exist_ok=True)

    print("🧠 Đang khởi tạo LightRAG kết nối Ollama...")
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="qwen2.5:7b",
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

    # --- KHU VỰC ĐỌC FILE CHECKPOINT (KIỂM TRA TIẾN TRÌNH CŨ) ----
    checkpoint_file = os.path.join(WORKING_DIR, "resume_checkpoint.txt")
    start_chunk_index = 0 # Mặc định chạy từ đầu (chunk 0)

    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                start_chunk_index = int(f.read().strip())
            print(f"🔄 Tìm thấy tiến trình cũ! Sẽ tiếp tục chạy từ chunk thứ: {start_chunk_index + 1}")
        except Exception:
            print("⚠️ File checkpoint bị lỗi, sẽ chạy lại từ đầu cho an toàn.")

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
                if content.strip(): docs_text.append(content)
            elif ext in [".txt", ".md"]:
                content = TextLoader(path, encoding="utf-8").load()[0].page_content
                if content.strip(): docs_text.append(content)
        except Exception as e:
            print(f"⚠️ Lỗi khi đọc file {path_obj.name}: {e}")

    if not docs_text:
        print("❌ Không có dữ liệu văn bản nào!")
        return

    # Chia nhỏ văn bản thành các chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text("\n\n".join(docs_text))
    
    total_chunks = len(chunks)
    print(f"⚡ Tổng số phân đoạn cần xử lý: {total_chunks}")

    # Bỏ qua các chunk đã xử lý thành công lần trước
    if start_chunk_index >= total_chunks:
        print("🎉 Toàn bộ dữ liệu đã được xử lý xong từ trước rồi!")
        return
        
    # --- VÒNG LẶP XỬ LÝ (CHỈ CHẠY TỪ START_CHUNK_INDEX) ---
    for i in range(start_chunk_index, total_chunks):
        chunk = chunks[i]
        print(f"\n🤖 Đang xử lý chunk {i+1}/{total_chunks}...")
        try:
            await rag.ainsert(chunk)
            
            # Ghi nhận đã xử lý xong chunk này vào file checkpoint
            with open(checkpoint_file, "w") as f:
                f.write(str(i + 1)) # Lần sau mở lên sẽ chạy từ chunk (i + 1)
                
            await asyncio.sleep(0.5) 
        except Exception as e:
            print(f"❌ Lỗi tại chunk {i+1}: {e}")
            print("⚠️ Hệ thống tạm dừng. Hãy chạy lại script sau khi sửa lỗi hoặc có điện lại.")
            return # Dừng chương trình để không ghi đè checkpoint sai
            
    # Xóa file checkpoint đi nếu đã hoàn thành 100% để lần sau nạp data mới không bị lẫn
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    # Đóng gói sản phẩm cuối cùng
    zip_output_dir = "/content/chatbot/lightrag_snapshots"
    os.makedirs(zip_output_dir, exist_ok=True)
    final_zip_path = os.path.join(zip_output_dir, "lightrag_final_backup")
    
    print("\n📦 Đang đóng gói toàn bộ đồ thị tri thức...")
    shutil.make_archive(final_zip_path, 'zip', WORKING_DIR)
    print(f"✅ Hoàn tất nạp dữ liệu! File của bạn: {final_zip_path}.zip")

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