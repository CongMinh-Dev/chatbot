# -*- coding: utf-8 -*-
import os
import glob
import requests
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 🔴 THAY ĐƯỜNG LINK NGROK BẠN NHẬN ĐƯỢC TỪ COLAB VÀO ĐÂY
url_ngrok="https://5535-35-240-158-30.ngrok-free.app/"
API_URL = f"{url_ngrok}api/ingest"
EXPORT_URL = f"{url_ngrok}api/export"

def main():
    print("📂 Quét tài liệu tại Local...")
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

    # Cắt nhỏ văn bản thành các chunks
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_text("\n\n".join(docs_text))
    
    print(f"⚡ Đang gửi {len(chunks)} phân đoạn lên Google Colab (GPU) để trích xuất Đồ thị...")
    
    with tqdm(total=len(chunks), desc="🤖 Đang chuyển dữ liệu") as pbar:
        for chunk in chunks:
            try:
                # Gửi request POST lên API Colab
                response = requests.post(API_URL, json={"chunk": chunk})
                if response.status_code != 200:
                    print(f"\n⚠️ Lỗi xử lý chunk: {response.text}")
            except Exception as e:
                print(f"\n⚠️ Lỗi kết nối tới Colab: {e}")
            pbar.update(1)
            
    print("\n✅ Đã gửi toàn bộ dữ liệu! Đang yêu cầu Colab đóng gói Đồ thị...")
    
    # Gọi API bắt Colab đóng gói zip thư mục DB lại
    res = requests.get(EXPORT_URL)
    print(res.json().get("message", "Xong! Hãy lên giao diện Colab tải file 'lightrag_db_exported.zip' về."))

if __name__ == "__main__":
    main()