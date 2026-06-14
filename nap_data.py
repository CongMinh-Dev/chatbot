import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# Thay đổi: Import OllamaEmbeddings thay vì OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

def ingest_data():
    documents = []
    folder_path = "./data"
    
    if not os.path.exists(folder_path):
        print(f"Thư mục '{folder_path}' không tồn tại.")
        return

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        try:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())
        except Exception as e:
            print(f"Lỗi khi xử lý file {file}: {e}")
    
    if not documents:
        print("Không tìm thấy file hợp lệ.")
        return

    MARKDOWN_SEPARATORS = [
        "\n#{1,6} ", "```\n", "\n\\*\\*\\*+\n", "\n---+\n", 
        "\n___+\n", "\n\n", "\n", " ", "",
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS
    )
    
    splits = text_splitter.split_documents(documents)
    
    # Cấu hình Embedding sử dụng Ollama với bge-m3
    print("Đang khởi tạo OllamaEmbeddings (bge-m3)...")
    embeddings = OllamaEmbeddings(model="bge-m3:latest", base_url="http://192.168.1.100:11434")
    
    # Khởi tạo và lưu vào ChromaDB
    print("Đang tạo vector store và lưu vào ./chroma_db...")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print("Đã lưu chỉ mục vector vào ChromaDB thành công!")

if __name__ == '__main__':
    ingest_data()