# -*- coding: utf-8 -*-
import os
import shutil
import asyncio
import logging  # <-- Thư viện để quản lý và bắt log bằng chữ
from pathlib import Path

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
from langchain_community.document_loaders import PyPDFLoader

WORKING_DIR = "/content/chatbot/lightrag_db"

# Khai báo biến toàn cục để bộ bắt log có thể dùng chung với hàm main
LOGGED_PERSIST_SUCCESS = False

# ========================================================
# 🤖 BỘ PHÁT HIỆN LOG GHI ĐĨA (CUSTOM LOG HANDLER)
# ========================================================
class LightRAGLogFilter(logging.Handler):
    def emit(self, record):
        global LOGGED_PERSIST_SUCCESS
        log_message = record.getMessage()
        
        # Bắt từ khóa ghi đĩa thành công của LightRAG
        if "In memory DB persist to disk" in log_message:
            LOGGED_PERSIST_SUCCESS = True

# Cấu hình log của hệ thống để add bộ lọc phía trên vào
logger = logging.getLogger()
logger.setLevel(logging.INFO)
custom_handler = LightRAGLogFilter()
logger.addHandler(custom_handler)
# ========================================================


# ========================================================
# 🛠️ KHỞI TẠO CẤU HÌNH LIGHTRAG TỐI ƯU SIÊU NHẸ CHO CPU
# ========================================================
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="qwen2.5:14b",            # Tên model LLM chạy trên Colab
    embedding_func=ollama_embed, 
    
    # ─── CẤU HÌNH BĂM NHỎ THEO Ý BẠN ĐỂ CPU CHẠY MƯỢT ───
    chunk_token_size=500,         # ~500 ký tự (Giúp CPU local xử lý prompt rất nhẹ và nhanh)
    chunk_overlap_token_size=100, # Chồng lấn 100 token giữ ngữ cảnh nối trang, chống lỗi 0 chunk
    
    addon_params={
        "language": "Vietnamese",
        "llm_model_kwargs": {"options": {"num_ctx": 8192}}, # Mở rộng ngữ cảnh khi xử lý trên GPU Colab
        "embedding_model_name": "embeddinggemma:latest" 
    }
)


# ========================================================
# 🚀 HÀM CHÍNH ĐỌC PDF NGUYÊN KHỐI VÀ NẠP VÀO GRAPH_RAG
# ========================================================
async def main():
    global LOGGED_PERSIST_SUCCESS
    
    file_path = "/content/chatbot/data/foodPet.pdf"
    
    if not os.path.exists(file_path):
        print(f"❌ Sai đường dẫn! Không tìm thấy file dữ liệu tại: {file_path}")
        return

    # 1. Đọc toàn bộ nội dung text từ file PDF gốc
    print("📄 Đang đọc toàn bộ dữ liệu từ file PDF...")
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # Ghép tất cả văn bản các trang lại thành 1 chuỗi lớn liên tục độc nhất
    full_pdf_text = "\n".join([doc.page_content for doc in docs])
    print(f"✅ Đọc thành công! Tổng số ký tự thô: {len(full_pdf_text)}")

    # 2. Khởi tạo kho lưu trữ rỗng mới
    await rag.initialize_storages()

    print("\n🚀 Đang tiến hành nạp NGUYÊN KHỐI văn bản liên tục vào LightRAG...")
    print("💡 Hệ thống sẽ tự động băm nhỏ thông minh thành các chunk ~350 token.")
    print("⏳ Đang chạy trích xuất thực thể đồ thị Tri thức (Vui lòng đợi)...")
    
    # Reset lại cờ bắt log ghi đĩa trước khi nạp
    LOGGED_PERSIST_SUCCESS = False
    
    try:
        # BỎ HOÀN TOÀN vòng lặp 'for chunk' cũ. Nạp trực tiếp cả khối văn bản lớn.
        await rag.ainsert(full_pdf_text)
        
        # Kiểm tra tính an toàn dữ liệu thông qua Custom Log Handler
        if LOGGED_PERSIST_SUCCESS:
            print("💾 Log hệ thống: Đã xác nhận cơ sở dữ liệu đồ thị lưu xuống đĩa thành công!")
        else:
            print("⚠️ [Cảnh báo] Lệnh ainsert đã chạy xong nhưng hệ thống chưa kích hoạt lệnh Persist Disk.")
            
    except Exception as e:
        print(f"❌ Lỗi crash trong quá trình xử lý nạp dữ liệu: {e}")
        return

    # 3. Tiến hành đóng gói nén zip thư mục lightrag_db để bạn tải về máy cá nhân CPU
    zip_output_path = "/content/chatbot/lightrag_snapshots/lightrag_final_backup"
    os.makedirs("/content/chatbot/lightrag_snapshots", exist_ok=True)
    
    print("\n📦 Đang đóng gói toàn bộ thư mục đồ thị tri thức siêu nhẹ...")
    shutil.make_archive(zip_output_path, 'zip', WORKING_DIR)
    print(f"🎯 HOÀN TẤT THÀNH CÔNG! File sao lưu của bạn đã sẵn sàng tại:\n👉 {zip_output_path}.zip")

if __name__ == '__main__':
    asyncio.run(main())