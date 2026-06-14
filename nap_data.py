import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

def ingest_data():
    documents = []
    folder_path = "./data"
    
    if not os.path.exists(folder_path):
        print(f"Thư mục '{folder_path}' không tồn tại.")
        return

    # 1. Load tài liệu
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

    # 2. Định nghĩa separators giữ nguyên như yêu cầu
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
    print(f"Đã chia nhỏ thành {len(splits)} đoạn. Bắt đầu tạo embedding...")

    # 3. Khởi tạo Embedding
    # đây là IP LXC thực tế của tôi
    embeddings = OllamaEmbeddings(
        model="bge-m3:latest", 
        base_url="http://192.168.1.100:11434"
    )
    
    # 4. Khởi tạo VectorStore và lưu thủ công từng đoạn để tránh treo
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    for i, split in enumerate(splits):
        try:
            print(f"Đang xử lý đoạn {i+1}/{len(splits)}...")
            vectorstore.add_documents([split])
        except Exception as e:
            print(f"Lỗi tại đoạn {i+1}: {e}")
            # Nếu lỗi, có thể dừng lại hoặc tiếp tục tùy bạn
    
    print("Đã lưu chỉ mục vector vào ChromaDB thành công!")

if __name__ == '__main__':
    ingest_data()