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
CHECKPOINT_FILE = os.path.join(WORKING_DIR, "ingest_checkpoint.txt")
ZIP_OUTPUT_PATH = "/content/chatbot/lightrag_db_exported"
SAVE_EVERY_N_CHUNKS = 3  # Cấu hình số lượng chunk để nén file định kỳ

@wrap_embedding_func_with_attrs(
    embedding_dim=768,
    max_token_size=8192,
    model_name="embeddinggemma"
)
async def custom_ollama_embed(texts, **kwargs):
    return await ollama_embed(texts, model="embeddinggemma", **kwargs)

def save_progress_and_zip(current_index):
    """Ghi nhận vị trí đã xử lý và tiến hành đóng gói dữ liệu"""
    # Ghi lại index tiếp theo cần xử lý vào file checkpoint
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        f.write(str(current_index))
    
    # Nén thư mục dữ liệu lại thành file zip khẩn cấp
    shutil.make_archive(ZIP_OUTPUT_PATH, 'zip', WORKING_DIR)
    print(f"\n💾 Đã lưu checkpoint tại chunk {current_index} và cập nhật file: {ZIP_OUTPUT_PATH}.zip")

async def main():
    # BỎ đoạn rmtree để giữ lại dữ liệu cũ khi Colab bị sập và chạy lại
    os.makedirs(WORKING_DIR, exist_ok=True)

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
    
    # Hàm này tự động nạp lại các file đồ thị cũ (.json, .nano-graph...) từ WORKING_DIR nếu chúng đã tồn tại
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

    # Đọc toàn bộ tài liệu (Đảm bảo thứ tự đọc file đồng nhất)
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

    # Tiến hành chia nhỏ văn bản thành các chunks 500 ký tự
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text("\n\n".join(docs_text))
    
    total_chunks = len(chunks)
    print(f"⚡ Tổng số phân đoạn tìm thấy: {total_chunks}")
    
    if start_chunk_index >= total_chunks:
        print("✅ Toàn bộ các chunk đã được xử lý xong từ trước đó!")
        return

    # Chỉ xử lý các chunk chưa chạy
    remaining_chunks = chunks[start_chunk_index:]
    print(f"🚀 Bắt đầu xử lý tiếp từ chunk {start_chunk_index + 1}/{total_chunks}...")

    # Nạp trực tiếp từng phân đoạn vào LightRAG thông qua cơ chế bất đồng bộ
    for i, chunk in enumerate(tqdm(remaining_chunks, desc="🤖 Đang xử lý")):
        actual_index = start_chunk_index + i
        try:
            await rag.ainsert(chunk)
            
            # Kiểm tra định kỳ (Ví dụ: chạy xong cứ mỗi 3 chunks thì lưu trạng thái & nén)
            if (i + 1) % SAVE_EVERY_N_CHUNKS == 0:
                # Ép LightRAG ghi toàn bộ dữ liệu từ bộ nhớ đệm xuống đĩa cứng trước khi nén
                await rag.kv_storage.index_done_callback()
                save_progress_and_zip(actual_index + 1)
                
        except Exception as e:
            print(f"\n❌ Lỗi tại chunk {actual_index + 1}: {e}")
            # Nếu lỗi, cố gắng sao lưu lại phần an toàn trước đó
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