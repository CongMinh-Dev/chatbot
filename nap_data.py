import os
import json
import time
import requests
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

load_dotenv()

# --- BIẾN MÔI TRƯỜNG ---
POSTGRES_DB = os.getenv("POSTGRES_DB", "crm_db")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

embeddings = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", api_key=NVIDIA_API_KEY)
redis_client = redis.Redis.from_url(REDIS_URL)

def get_db_connection():
    return psycopg2.connect(
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT
    )

def process_word_file(file_path: str, id_ho_so_khach: str, id_kenh: str):
    """Đọc file Word và lưu vector vào postgres"""
    if not os.path.exists(file_path):
        print(f"❌ File không tồn tại: {file_path}")
        return

    try:
        loader = Docx2txtLoader(file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = text_splitter.split_documents(documents)

        conn = get_db_connection()
        cur = conn.cursor()

        # Xóa vector cũ của khách này (nếu nạp lại)
        cur.execute(
            "DELETE FROM file_khach_hang_embeddings WHERE id_ho_so_khach = %s", 
            (id_ho_so_khach,)
        )

        for doc in splits:
            content = doc.page_content
            vector = embeddings.embed_query(content)
            
            cur.execute("""
                INSERT INTO file_khach_hang_embeddings (id_kenh, id_ho_so_khach, noi_dung, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s)
            """, (id_kenh, id_ho_so_khach, content, vector, json.dumps({"file_path": file_path})))

        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ Đã nạp thành công {len(splits)} đoạn vector cho khách hàng {id_ho_so_khach}")

        # Xóa file tạm sau khi nạp xong
        if os.path.exists(file_path):
            os.remove(file_path)

    except Exception as e:
        print(f"❌ Lỗi nạp file Word: {e}")

def sync_woocommerce_categories(id_kenh: str):
    """Đồng bộ danh mục WooCommerce từ API vào Postgres & Vectorize"""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    cur.execute("SELECT domain_website, token_truy_cap, token_lam_moi FROM kenh_ket_noi WHERE id = %s", (id_kenh,))
    kenh = cur.fetchone()
    if not kenh or not kenh['domain_website']:
        print("❌ Kênh không tồn tại hoặc thiếu cấu hình WooCommerce URL")
        return

    woo_url = f"{kenh['domain_website'].rstrip('/')}/wp-json/wc/v3/products/categories"
    
    try:
        res = requests.get(
            woo_url, 
            auth=(kenh['token_truy_cap'], kenh['token_lam_moi']), 
            params={"per_page": 100}, 
            timeout=10
        )
        if res.status_code == 200:
            categories = res.json()
            for cat in categories:
                cat_id_ngoai = str(cat.get("id"))
                name = cat.get("name", "")
                slug = cat.get("slug", "")
                desc = cat.get("description", "")

                # 1. Upsert vào bảng danh_muc_san_pham
                cur.execute("""
                    INSERT INTO danh_muc_san_pham (id_kenh, id_danh_muc_ngoai, ten_danh_muc, slug, mo_ta)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id_kenh, id_danh_muc_ngoai) 
                    DO UPDATE SET ten_danh_muc = EXCLUDED.ten_danh_muc, slug = EXCLUDED.slug, mo_ta = EXCLUDED.mo_ta
                    RETURNING id;
                """, (id_kenh, cat_id_ngoai, name, slug, desc))
                
                db_cat_id = cur.fetchone()['id']

                # 2. Vectorize danh mục để AI search
                text_content = f"Danh mục sản phẩm: {name}. Slug: {slug}. Mô tả: {desc}"
                cat_vector = embeddings.embed_query(text_content)

                cur.execute("DELETE FROM danh_muc_embeddings WHERE id_danh_muc = %s", (db_cat_id,))
                cur.execute("""
                    INSERT INTO danh_muc_embeddings (id_danh_muc, id_kenh, noi_dung, embedding)
                    VALUES (%s, %s, %s, %s)
                """, (db_cat_id, id_kenh, text_content, cat_vector))

            conn.commit()
            print(f"✅ Đồng bộ thành công {len(categories)} danh mục Woo cho kênh {id_kenh}")
    except Exception as e:
        print(f"❌ Lỗi đồng bộ danh mục Woo: {e}")
    finally:
        cur.close()
        conn.close()

def run_nap_data_worker():
    """Vòng lặp lắng nghe Job từ Redis Queue"""
    print("🚀 Worker nap_data đang chạy và lắng nghe Redis Queue 'ingest_queue'...")
    while True:
        try:
            # BLPOP bắt worker ngủ chờ cho đến khi có job mới từ Node.js
            packed = redis_client.blpop("ingest_queue", timeout=30)
            if packed:
                _, data_bytes = packed
                job = json.loads(data_bytes.decode('utf-8'))
                
                job_type = job.get("type")
                if job_type == "word":
                    process_word_file(job["file_path"], job["id_ho_so_khach"], job["id_kenh"])
                elif job_type == "sync_woo":
                    sync_woocommerce_categories(job["id_kenh"])
        except Exception as e:
            print(f"⚠️ Lỗi Worker Ingest: {e}")
            time.sleep(1)

if __name__ == "__main__":
    run_nap_data_worker()