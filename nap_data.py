import os
import time
import requests
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from dotenv import load_dotenv

load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
CONSUMER_KEY_ENV = os.getenv("CONSUMER_KEY_ENV")
CONSUMER_SECRET_ENV = os.getenv("CONSUMER_SECRET_ENV")

def ingest_file_data(embeddings):
    """1. Xử lý nạp dữ liệu từ file tĩnh (.pdf, .docx)"""
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
        print("Không tìm thấy file tĩnh hợp lệ để nạp.")
        return

    MARKDOWN_SEPARATORS = [
        "\n# ", "\n## ", "\n### ", "```\n", 
        "\n\\*\\*\\*+\n", "\n---+\n", "\n___+\n", 
        "\n\n", "\n", " ", ""
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS
    )
    
    splits = text_splitter.split_documents(documents)
    print(f"Đã chia nhỏ tài liệu thành {len(splits)} đoạn. Tiến hành nạp vào ChromaDB...")

    # Gán collection_name riêng biệt cho nội dung RAG tĩnh
    vectorstore = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embeddings,
        collection_name="rag_contents"
    )
    
    for i, split in enumerate(splits):
        try:
            vectorstore.add_documents([split])
        except Exception as e:
            print(f"Lỗi tại đoạn file tĩnh {i+1}: {e}")
    
    print("--- Hoàn thành nạp dữ liệu FILE TĨNH! ---")

def ingest_woocommerce_categories(embeddings):
    """2. Xử lý nạp dữ liệu danh mục tự động từ WooCommerce API"""
    print("Bắt đầu quét danh mục từ WooCommerce...")
    WOO_CAT_URL = "https://minhshop.minh2309.io.vn/wp-json/wc/v3/products/categories"
    CONSUMER_KEY = CONSUMER_KEY_ENV
    CONSUMER_SECRET = CONSUMER_SECRET_ENV
    
    all_categories = []
    page = 1
    
    try:
        while True:
            response = requests.get(
                WOO_CAT_URL, 
                auth=(CONSUMER_KEY, CONSUMER_SECRET), 
                params={"per_page": 100, "page": page}, 
                timeout=5
            )
            if response.status_code == 200:
                cats = response.json()
                if not cats:
                    break
                all_categories.extend(cats)
                if len(cats) < 100:
                    break
                page += 1
            else:
                break
        
        if not all_categories:
            print("Không tìm thấy danh mục nào trên hệ thống WooCommerce.")
            return

        # Kết nối collection danh mục riêng biệt
        vectorstore_cat = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings,
            collection_name="woocommerce_categories"
        )

        # Xóa dữ liệu danh mục cũ để ghi đè dữ liệu mới nhất sạch sẽ
        try:
            existing_data = vectorstore_cat.get()
            if existing_data and existing_data["ids"]:
                vectorstore_cat.delete(ids=existing_data["ids"])
        except Exception:
            pass

        texts = []
        metadatas = []
        ids = []

        for cat in all_categories:
            cat_id = str(cat.get("id"))
            cat_name = cat.get("name", "")
            cat_slug = cat.get("slug", "")
            cat_description = cat.get("description", "")

            text_content = f"Danh mục sản phẩm: {cat_name}. Đường dẫn slug: {cat_slug}."
            if cat_description:
                text_content += f" Mô tả nhóm ngành hàng: {cat_description}"

            texts.append(text_content)
            metadatas.append({"id": cat_id, "name": cat_name, "slug": cat_slug})
            ids.append(cat_id)

        # Tiến hành nạp vector danh mục
        vectorstore_cat.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        print(f"--- Hoàn thành nạp {len(texts)} DANH MỤC từ WooCommerce vào Vector DB! ---")

    except Exception as e:
        print(f"Lỗi khi đồng bộ danh mục WooCommerce: {e}")

if __name__ == '__main__':
    # Khởi tạo embedding dùng chung
    embeddings = NVIDIAEmbeddings(
        model="nvidia/nv-embed-v1", 
        api_key=NVIDIA_API_KEY
    )
    
    # Chạy đồng thời 2 tiến trình nạp
    ingest_file_data(embeddings)
    ingest_woocommerce_categories(embeddings)
    print("Mọi chỉ mục Vector đã được lưu trữ thành công!")