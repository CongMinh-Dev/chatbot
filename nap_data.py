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



# ========================================================
# CHỈNH SỬA VỊ TRÍ 2: Đọc nguyên khối PDF và đẩy thẳng vào LightRAG
# ========================================================
async def main():
    global CURRENT_CHUNK_INDEX
    
    # Đường dẫn file PDF của bạn (giữ nguyên PDF, không cần đổi sang .txt)
    file_path = "/content/chatbot/data/foodPet.pdf" 
    
    if not os.path.exists(file_path):
        print(f"❌ Không tìm thấy file dữ liệu tại: {file_path}")
        return

    # 1. Đọc toàn bộ file PDF và ghép các trang thành 1 chuỗi văn bản duy nhất
    print("📄 Đang đọc dữ liệu từ file PDF...")
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # Ghép toàn bộ nội dung text của các trang lại với nhau, giữ nguyên tính liên tục
    full_pdf_text = "\n".join([doc.page_content for doc in docs])
    print(f"✅ Đọc thành công! Tổng số ký tự thô: {len(full_pdf_text)}")

    # 2. Khởi tạo kho lưu trữ đồ thị
    await rag.initialize_storages()

    print("\n🚀 Đang tiến hành nạp NGUYÊN KHỐI văn bản vào LightRAG...")
    print("💡 LightRAG sẽ tự động băm thông minh theo cấu hình chunk_token_size=350.")
    print("⏳ Quá trình trích xuất thực thể đồ thị bắt đầu (Vui lòng đợi)...")
    
    try:
        # BỎ HOÀN TOÀN vòng lặp for cũ. Đẩy nguyên khối text vào bằng 1 lệnh duy nhất.
        await rag.ainsert(full_pdf_text)
        print("✅ Thao tác kết thúc lệnh ainsert thành công!")
    except Exception as e:
        print(f"❌ Lỗi crash trong quá trình nạp dữ liệu: {e}")
        return

    # 3. Đóng gói sao lưu kết quả sau khi hoàn thành
    zip_output_path = "/content/chatbot/lightrag_snapshots/lightrag_final_backup"
    os.makedirs("/content/chatbot/lightrag_snapshots", exist_ok=True)
    print("\n📦 Đang đóng gói toàn bộ đồ thị tri thức mới...")
    shutil.make_archive(zip_output_path, 'zip', WORKING_DIR)
    print(f"🎯 HOÀN TẤT! Đồ thị siêu nhẹ cho CPU đã được tạo thành công tại: {zip_output_path}.zip")

if __name__ == '__main__':
    asyncio.run(main())