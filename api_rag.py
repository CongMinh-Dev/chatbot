import os
import json
import asyncio
import redis.asyncio as aioredis
import asyncpg
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_deepseek import ChatDeepSeek
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- BIẾN MÔI TRƯỜNG ---
POSTGRES_DB = os.getenv("POSTGRES_DB", "crm_db")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# --- PROMPTS ---
ANALYSIS_PROMPT = """
Bạn là bộ phân tích ý định khách hàng E-commerce.
Đọc "Lịch sử hội thoại" và "Câu hỏi mới nhất" để làm 2 bước:
1. standalone_question: Viết lại câu hỏi đầy đủ ngữ cảnh.
2. target & filters: 
   - Nếu cần tra kho WooCommerce (giá, hàng còn không, biến thể, tìm danh sách...): "target": "woocommerce"
   - Nếu hỏi đáp chung (chính sách, tư vấn size, file tài liệu cá nhân): "target": "rag"

Trả về đúng 1 chuỗi JSON duy nhất, KHÔNG kèm markdown:
{{
  "standalone_question": "string",
  "target": "woocommerce" hoặc "rag",
  "filters": {{
    "max_price": null,
    "min_price": null,
    "stock_check": false,
    "category": null,
    "page": 1
  }}
}}

Lịch sử:
{history}
Câu hỏi:
{question}
"""

SALES_PROMPT = """
Bạn là nhân viên bán hàng chuyên nghiệp (luôn xưng dạ, em).
Chỉ trả lời dựa vào thông tin tài liệu dưới đây. Nếu không có trả lời: 'Dạ để em hỏi lại sếp'.
Tài liệu:
{context}
Câu hỏi:
{question}
"""

API_RESPONSE_PROMPT = """
Bạn là nhân viên bán hàng chuyên nghiệp (luôn xưng dạ, em).
Dựa vào dữ liệu sản phẩm WooCommerce dưới đây để trả lời CỰC KỲ NGẮN GỌN.
Dữ liệu:
{api_context}
Câu hỏi:
{question}
"""

class RAGService:
    def __init__(self):
        self.redis = None
        self.pg_pool = None
        self.embeddings = None
        self.llm = None
        self.httpx_client = None

    async def init_resources(self):
        self.redis = await aioredis.from_url(REDIS_URL)
        self.pg_pool = await asyncpg.create_pool(
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            min_size=5,
            max_size=25
        )
        self.httpx_client = httpx.AsyncClient(timeout=10.0)
        self.embeddings = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", api_key=NVIDIA_API_KEY)
        self.llm = ChatDeepSeek(model="deepseek-v4-flash", api_key=DEEPSEEK_API_KEY, temperature=0.1)

    async def close_resources(self):
        if self.redis: await self.redis.close()
        if self.pg_pool: await self.pg_pool.close()
        if self.httpx_client: await self.httpx_client.aclose()

    # --- NẠP DỮ LIỆU FILE WORD (ASYNC) ---
    async def process_word_file(self, file_path: str, id_ho_so_khach: str, id_kenh: str):
        if not os.path.exists(file_path):
            print(f"❌ File không tồn tại: {file_path}")
            return

        try:
            loader = Docx2txtLoader(file_path)
            documents = await asyncio.to_thread(loader.load)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            splits = text_splitter.split_documents(documents)

            async with self.pg_pool.acquire() as conn:
                await conn.execute("DELETE FROM file_khach_hang_embeddings WHERE id_ho_so_khach = $1", id_ho_so_khach)

                for doc in splits:
                    content = doc.page_content
                    vector = await self.embeddings.aembed_query(content)
                    await conn.execute("""
                        INSERT INTO file_khach_hang_embeddings (id_kenh, id_ho_so_khach, noi_dung, embedding, metadata)
                        VALUES ($1, $2, $3, $4, $5)
                    """, id_kenh, id_ho_so_khach, content, str(vector), json.dumps({"file_path": file_path}))

            print(f"✅ Nạp {len(splits)} đoạn vector cho khách {id_ho_so_khach}")
            if os.path.exists(file_path): os.remove(file_path)
        except Exception as e:
            print(f"❌ Lỗi nạp file Word: {e}")

    # --- ĐỒNG BỘ DANH MỤC WOOCOMMERCE (ASYNC) ---
    async def sync_woocommerce_categories(self, id_kenh: str):
        async with self.pg_pool.acquire() as conn:
            kenh = await conn.fetchrow("SELECT domain_website, token_truy_cap, token_lam_moi FROM kenh_ket_noi WHERE id = $1", id_kenh)
            if not kenh or not kenh['domain_website']: return

            woo_url = f"{kenh['domain_website'].rstrip('/')}/wp-json/wc/v3/products/categories"
            try:
                res = await self.httpx_client.get(
                    woo_url, 
                    auth=(kenh['token_truy_cap'], kenh['token_lam_moi']),
                    params={"per_page": 100}
                )
                if res.status_code == 200:
                    categories = res.json()
                    for cat in categories:
                        cat_id_ngoai = str(cat.get("id"))
                        name, slug, desc = cat.get("name", ""), cat.get("slug", ""), cat.get("description", "")

                        db_cat_id = await conn.fetchval("""
                            INSERT INTO danh_muc_san_pham (id_kenh, id_danh_muc_ngoai, ten_danh_muc, slug, mo_ta)
                            VALUES ($1, $2, $3, $4, $5)
                            ON CONFLICT (id_kenh, id_danh_muc_ngoai) 
                            DO UPDATE SET ten_danh_muc = EXCLUDED.ten_danh_muc, slug = EXCLUDED.slug, mo_ta = EXCLUDED.mo_ta
                            RETURNING id;
                        """, id_kenh, cat_id_ngoai, name, slug, desc)

                        text_content = f"Danh mục sản phẩm: {name}. Slug: {slug}. Mô tả: {desc}"
                        cat_vector = await self.embeddings.aembed_query(text_content)

                        await conn.execute("DELETE FROM danh_muc_embeddings WHERE id_danh_muc = $1", db_cat_id)
                        await conn.execute("""
                            INSERT INTO danh_muc_embeddings (id_danh_muc, id_kenh, noi_dung, embedding)
                            VALUES ($1, $2, $3, $4)
                        """, db_cat_id, id_kenh, text_content, str(cat_vector))
                    print(f"✅ Đồng bộ {len(categories)} danh mục Woo kênh {id_kenh}")
            except Exception as e:
                print(f"❌ Lỗi đồng bộ danh mục Woo: {e}")

    # --- XỬ LÝ TIN NHẮN RAG & CHAT ---
    async def get_conversation_history(self, conn, id_cuoc_hoi_thoai):
        rows = await conn.fetch("""
            SELECT loai_nguoi_gui, noi_dung 
            FROM tin_nhan WHERE id_cuoc_hoi_thoai = $1 
            ORDER BY ngay_tao DESC LIMIT 5
        """, id_cuoc_hoi_thoai)
        return "\n".join([f"{'Khách hàng' if r['loai_nguoi_gui']=='khach_hang' else 'Bot'}: {r['noi_dung']}" for r in reversed(rows)])

    async def search_category_vector(self, conn, id_kenh, category_name):
        query_vec = await self.embeddings.aembed_query(category_name)
        row = await conn.fetchrow("""
            SELECT d.id_danh_muc_ngoai FROM danh_muc_embeddings e
            JOIN danh_muc_san_pham d ON e.id_danh_muc = d.id
            WHERE e.id_kenh = $1
            ORDER BY e.embedding <-> $2::vector LIMIT 1;
        """, id_kenh, str(query_vec))
        return row['id_danh_muc_ngoai'] if row else None

    async def query_woocommerce(self, conn, id_kenh, filters):
        row = await conn.fetchrow("SELECT domain_website, token_truy_cap, token_lam_moi FROM kenh_ket_noi WHERE id = $1", id_kenh)
        if not row or not row['domain_website']: return "Không tìm thấy cấu hình Woo."

        woo_url = f"{row['domain_website'].rstrip('/')}/wp-json/wc/v3/products"
        params = {"status": "publish", "per_page": 5, "page": filters.get("page", 1)}
        
        if filters.get("category"):
            cat_id = await self.search_category_vector(conn, id_kenh, filters["category"])
            if cat_id: params["category"] = cat_id
            else: params["search"] = filters["category"]

        try:
            res = await self.httpx_client.get(woo_url, params=params, auth=(row['token_truy_cap'], row['token_lam_moi']))
            if res.status_code == 200:
                return "Danh sách sản phẩm:\n" + "\n".join([f"- Tên: {p.get('name')} | Giá: {p.get('price')}đ | Link: {p.get('permalink')}" for p in res.json()])
        except Exception as e:
            print(f"❌ Lỗi Woo: {e}")
        return "Lỗi kết nối Woo."

    async def search_rag_context(self, conn, id_kenh, id_ho_so_khach, question):
        query_vec = await self.embeddings.aembed_query(question)
        rows_word = await conn.fetch("SELECT noi_dung FROM file_khach_hang_embeddings WHERE id_ho_so_khach = $1 ORDER BY embedding <-> $2::vector LIMIT 2;", id_ho_so_khach, str(query_vec))
        rows_common = await conn.fetch("SELECT noi_dung FROM kho_tri_thuc_ai WHERE id_kenh = $1 ORDER BY vector_dac_trung <-> $2::vector LIMIT 2;", id_kenh, str(query_vec))
        docs = [r['noi_dung'] for r in rows_word] + [r['noi_dung'] for r in rows_common]
        return "\n\n".join(docs) if docs else "Không có tài liệu."

    async def process_message(self, id_tin_nhan):
        async with self.pg_pool.acquire() as conn:
            msg = await conn.fetchrow("""
                SELECT t.id, t.noi_dung, t.id_cuoc_hoi_thoai, c.id_kenh, c.id_ho_so_khach, c.bat_bot_tu_dong
                FROM tin_nhan t JOIN cuoc_hoi_thoai c ON t.id_cuoc_hoi_thoai = c.id
                WHERE t.id = $1
            """, id_tin_nhan)

            if not msg or not msg['bat_bot_tu_dong']: return

            question, id_cuoc_hoi_thoai, id_kenh, id_ho_so_khach = msg['noi_dung'], msg['id_cuoc_hoi_thoai'], msg['id_kenh'], msg['id_ho_so_khach']
            history_text = await self.get_conversation_history(conn, id_cuoc_hoi_thoai)

            analysis_chain = ChatPromptTemplate.from_template(ANALYSIS_PROMPT) | self.llm | StrOutputParser()
            raw_analysis = await analysis_chain.ainvoke({"history": history_text, "question": question})
            
            try:
                res_json = json.loads(raw_analysis.replace("```json", "").replace("```", "").strip())
            except Exception:
                res_json = {"standalone_question": question, "target": "rag", "filters": {}}

            standalone_question = res_json.get("standalone_question", question)
            target = res_json.get("target", "rag")

            if target == "woocommerce":
                api_context = await self.query_woocommerce(conn, id_kenh, res_json.get("filters", {}))
                answer = await (ChatPromptTemplate.from_template(API_RESPONSE_PROMPT) | self.llm | StrOutputParser()).ainvoke({"api_context": api_context, "question": standalone_question})
            else:
                rag_context = await self.search_rag_context(conn, id_kenh, id_ho_so_khach, standalone_question)
                answer = await (ChatPromptTemplate.from_template(SALES_PROMPT) | self.llm | StrOutputParser()).ainvoke({"context": rag_context, "question": standalone_question})

            await conn.execute("INSERT INTO tin_nhan (id_cuoc_hoi_thoai, loai_nguoi_gui, noi_dung, loai_tin_nhan) VALUES ($1, 'bot', $2, 'van_ban')", id_cuoc_hoi_thoai, answer)

rag_service = RAGService()

# --- BACKGROUND WORKERS (CHẠY CHUNG PROCESS) ---
async def start_chat_worker():
    print("🚀 Worker Chat AI đã kích hoạt...")
    while True:
        try:
            packed = await rag_service.redis.blpop("process_ai_queue", timeout=10)
            if packed:
                _, msg_id_bytes = packed
                asyncio.create_task(rag_service.process_message(msg_id_bytes.decode('utf-8')))
        except Exception as e:
            await asyncio.sleep(0.2)

async def start_ingest_worker():
    print("🚀 Worker Ingest (Word + Woo) đã kích hoạt...")
    while True:
        try:
            packed = await rag_service.redis.blpop("ingest_queue", timeout=10)
            if packed:
                _, data_bytes = packed
                job = json.loads(data_bytes.decode('utf-8'))
                if job.get("type") == "word":
                    asyncio.create_task(rag_service.process_word_file(job["file_path"], job["id_ho_so_khach"], job["id_kenh"]))
                elif job.get("type") == "sync_woo":
                    asyncio.create_task(rag_service.sync_woocommerce_categories(job["id_kenh"]))
        except Exception as e:
            await asyncio.sleep(0.5)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await rag_service.init_resources()
    t1 = asyncio.create_task(start_chat_worker())
    t2 = asyncio.create_task(start_ingest_worker())
    yield
    t1.cancel()
    t2.cancel()
    await rag_service.close_resources()

app = FastAPI(title="CRM AI Service", lifespan=lifespan)

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "crm-ai-service"}

class TestRAGRequest(BaseModel):
    id_kenh: str
    id_ho_so_khach: str
    question: str

@app.post("/api/v1/test-rag")
async def test_rag_endpoint(payload: TestRAGRequest):
    return {"status": "success", "message": "API sẵn sàng mở rộng sau này!"}