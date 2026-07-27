import os
import json
import asyncio
import redis.asyncio as aioredis
import asyncpg
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI
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
   - Nếu hỏi sản phẩm, giá cả, mẫu mã, tồn kho, tìm đồ, mua sắm: "target": "woocommerce"
   - Nếu hỏi đáp chung (chính sách đổi trả, tư vấn chọn size chung, tài liệu cá nhân): "target": "rag"

Trả về đúng 1 chuỗi JSON duy nhất, KHÔNG kèm markdown:
{{
  "standalone_question": "string",
  "target": "woocommerce" hoặc "rag",
  "filters": {{
    "keyword": null,
    "max_price": null,
    "min_price": null,
    "stock_check": false,
    "category": null
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
Dựa vào dữ liệu sản phẩm trong kho dưới đây để tư vấn CỰC KỲ NGẮN GỌN, thân thiện, kèm giá và thông tin rõ ràng.
Dữ liệu sản phẩm tìm thấy:
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
        # Tối ưu RAM: Giới hạn max_size = 25 connections cho FastAPI Worker
        self.pg_pool = await asyncpg.create_pool(
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            min_size=2,
            max_size=25
        )
        self.httpx_client = httpx.AsyncClient(timeout=30.0)
        self.embeddings = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", api_key=NVIDIA_API_KEY)
        self.llm = ChatDeepSeek(model="deepseek-v4-flash", api_key=DEEPSEEK_API_KEY, temperature=0.1)

    async def close_resources(self):
        if self.redis: await self.redis.close()
        if self.pg_pool: await self.pg_pool.close()
        if self.httpx_client: await self.httpx_client.aclose()

    # =========================================================================
    # 1. BỘ XỬ LÝ NẠP FILE WORD TỪ SHARED VOLUME
    # =========================================================================
    async def ingest_word_file(self, file_id: str, file_path: str, id_ho_so_khach: str):
        """Đọc file Word từ Shared Volume -> Chunking -> Embeddings -> Xóa Vector Cũ -> Lưu Vector Mới"""
        if not os.path.exists(file_path):
            print(f"❌ File không tồn tại tại đường dẫn Shared Volume: {file_path}")
            return

        try:
            # 1. Đọc file word bằng Docx2txtLoader
            loader = Docx2txtLoader(file_path)
            documents = loader.load()

            # 2. Cắt nhỏ văn bản (Chunking)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_documents(documents)

            async with self.pg_pool.acquire() as conn:
                # 3. Xóa các vector cũ của file_id này trong DB trước khi nạp mới
                await conn.execute("DELETE FROM file_khach_hang_embeddings WHERE file_id = $1", file_id)

                # 4. Embed & Insert từng chunk vào Postgres
                for chunk in chunks:
                    text_content = chunk.page_content
                    vector = await self.embeddings.aembed_query(text_content)
                    await conn.execute("""
                        INSERT INTO file_khach_hang_embeddings (file_id, id_ho_so_khach, noi_dung, embedding)
                        VALUES ($1, $2, $3, $4)
                    """, file_id, id_ho_so_khach, text_content, str(vector))

            print(f"✅ [SUCCESS] Đã Vector hóa thành công File Word ID: {file_id}. File vật lý được GIỮ NGUYÊN trên đĩa.")
        except Exception as e:
            print(f"❌ Lỗi xử lý Ingest Word: {e}")

    # =========================================================================
    # 2. BỘ ĐỒNG BỘ WOOCOMMERCE
    # =========================================================================
    async def sync_all_woocommerce_data(self, id_kenh: str):
        async with self.pg_pool.acquire() as conn:
            kenh = await conn.fetchrow("SELECT domain_website, token_truy_cap, token_lam_moi FROM kenh_ket_noi WHERE id = $1", id_kenh)
            if not kenh or not kenh['domain_website']: return

            base_url = kenh['domain_website'].rstrip('/')
            auth = (kenh['token_truy_cap'], kenh['token_lam_moi'])

            try:
                await self._sync_categories(conn, id_kenh, base_url, auth)
                await self._sync_products_and_variations(conn, id_kenh, base_url, auth)
                print(f"🎉 [SUCCESS] Đã đồng bộ hoàn tất dữ liệu Woo cho kênh {id_kenh}")
            except Exception as e:
                print(f"❌ Lỗi đồng bộ Woo: {e}")

    async def _sync_categories(self, conn, id_kenh, base_url, auth):
        url = f"{base_url}/wp-json/wc/v3/products/categories"
        res = await self.httpx_client.get(url, auth=auth, params={"per_page": 100})
        if res.status_code == 200:
            for cat in res.json():
                cat_id_ngoai = str(cat.get("id"))
                name, slug, desc = cat.get("name", ""), cat.get("slug", ""), cat.get("description", "")

                db_cat_id = await conn.fetchval("""
                    INSERT INTO danh_muc_san_pham (id_kenh, id_danh_muc_ngoai, ten_danh_muc, slug, mo_ta)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (id_kenh, id_danh_muc_ngoai) 
                    DO UPDATE SET ten_danh_muc = EXCLUDED.ten_danh_muc, slug = EXCLUDED.slug, mo_ta = EXCLUDED.mo_ta
                    RETURNING id;
                """, id_kenh, cat_id_ngoai, name, slug, desc)

                text_content = f"Danh mục: {name}. Mô tả: {desc}"
                cat_vector = await self.embeddings.aembed_query(text_content)
                await conn.execute("DELETE FROM danh_muc_embeddings WHERE id_danh_muc = $1", db_cat_id)
                await conn.execute("""
                    INSERT INTO danh_muc_embeddings (id_danh_muc, id_kenh, noi_dung, embedding)
                    VALUES ($1, $2, $3, $4)
                """, db_cat_id, id_kenh, text_content, str(cat_vector))

    async def _sync_products_and_variations(self, conn, id_kenh, base_url, auth):
        page = 1
        while True:
            url = f"{base_url}/wp-json/wc/v3/products"
            res = await self.httpx_client.get(url, auth=auth, params={"per_page": 50, "page": page})
            if res.status_code != 200 or not res.json():
                break

            products = res.json()
            for p in products:
                sp_id_ngoai = str(p.get("id"))
                ten_sp = p.get("name", "")
                gia = float(p.get("price") or 0)
                mo_ta = p.get("description", "") or p.get("short_description", "")
                ton_kho = p.get("stock_quantity") or 0
                link = p.get("permalink", "")
                type_sp = p.get("type", "simple")

                db_sp_id = await conn.fetchval("""
                    INSERT INTO san_pham (id_kenh, id_san_pham_ngoai, ten_san_pham, gia, mo_ta, ton_kho, link_san_pham, loai_san_pham)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (id_kenh, id_san_pham_ngoai) 
                    DO UPDATE SET ten_san_pham = EXCLUDED.ten_san_pham, gia = EXCLUDED.gia, 
                                  mo_ta = EXCLUDED.mo_ta, ton_kho = EXCLUDED.ton_kho, link_san_pham = EXCLUDED.link_san_pham
                    RETURNING id;
                """, id_kenh, sp_id_ngoai, ten_sp, gia, mo_ta, ton_kho, link, type_sp)

                var_text_list = []
                if type_sp == "variable":
                    var_url = f"{base_url}/wp-json/wc/v3/products/{sp_id_ngoai}/variations"
                    var_res = await self.httpx_client.get(var_url, auth=auth, params={"per_page": 50})
                    if var_res.status_code == 200:
                        for v in var_res.json():
                            v_id_ngoai = str(v.get("id"))
                            v_gia = float(v.get("price") or gia)
                            v_ton_kho = v.get("stock_quantity") or 0
                            attrs = ", ".join([f"{a.get('name')}: {a.get('option')}" for a in v.get("attributes", [])])
                            
                            await conn.execute("""
                                INSERT INTO bien_the_san_pham (id_san_pham, id_bien_the_ngoai, thuoc_tinh, gia, ton_kho)
                                VALUES ($1, $2, $3, $4, $5)
                                ON CONFLICT (id_san_pham, id_bien_the_ngoai)
                                DO UPDATE SET thuoc_tinh = EXCLUDED.thuoc_tinh, gia = EXCLUDED.gia, ton_kho = EXCLUDED.ton_kho
                            """, db_sp_id, v_id_ngoai, attrs, v_gia, v_ton_kho)
                            
                            var_text_list.append(f"Biến thể ({attrs}) - Giá: {v_gia}đ")

                text_to_embed = f"Sản phẩm: {ten_sp}. Giá: {gia}đ. Mô tả: {mo_ta}. " + " ".join(var_text_list)
                sp_vector = await self.embeddings.aembed_query(text_to_embed)

                await conn.execute("DELETE FROM san_pham_embeddings WHERE id_san_pham = $1", db_sp_id)
                await conn.execute("""
                    INSERT INTO san_pham_embeddings (id_san_pham, id_kenh, noi_dung, embedding)
                    VALUES ($1, $2, $3, $4)
                """, db_sp_id, id_kenh, text_to_embed, str(sp_vector))

            page += 1

    # =========================================================================
    # 3. LUỒNG TRUY VẤN
    # =========================================================================
    async def search_products_from_postgres(self, conn, id_kenh, question, filters):
        query_vec = await self.embeddings.aembed_query(question)
        rows = await conn.fetch("""
            SELECT s.ten_san_pham, s.gia, s.mo_ta, s.ton_kho, s.link_san_pham,
                   COALESCE(string_agg(concat(b.thuoc_tinh, ' (Giá: ', b.gia, 'đ)'), '; '), 'Không có biến thể') as ds_bien_the
            FROM san_pham_embeddings e
            JOIN san_pham s ON e.id_san_pham = s.id
            LEFT JOIN bien_the_san_pham b ON s.id = b.id_san_pham
            WHERE e.id_kenh = $1
            GROUP BY s.id, e.embedding
            ORDER BY e.embedding <-> $2::vector LIMIT 5;
        """, id_kenh, str(query_vec))

        if not rows: return "Không tìm thấy sản phẩm nào phù hợp trong kho."

        res_text = "Danh sách sản phẩm tìm thấy trong hệ thống:\n"
        for r in rows:
            res_text += f"- Tên: {r['ten_san_pham']} | Giá: {r['gia']}đ | Tồn kho: {r['ton_kho']} | Link: {r['link_san_pham']}\n"
            res_text += f"  Chi tiết mẫu mã/biến thể: {r['ds_bien_the']}\n"
        return res_text

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
            
            rows = await conn.fetch("SELECT loai_nguoi_gui, noi_dung FROM tin_nhan WHERE id_cuoc_hoi_thoai = $1 ORDER BY ngay_tao DESC LIMIT 5", id_cuoc_hoi_thoai)
            history_text = "\n".join([f"{'Khách hàng' if r['loai_nguoi_gui']=='khach_hang' else 'Bot'}: {r['noi_dung']}" for r in reversed(rows)])

            analysis_chain = ChatPromptTemplate.from_template(ANALYSIS_PROMPT) | self.llm | StrOutputParser()
            raw_analysis = await analysis_chain.ainvoke({"history": history_text, "question": question})
            
            try:
                res_json = json.loads(raw_analysis.replace("```json", "").replace("```", "").strip())
            except Exception:
                res_json = {"standalone_question": question, "target": "rag", "filters": {}}

            standalone_question = res_json.get("standalone_question", question)
            target = res_json.get("target", "rag")

            if target == "woocommerce":
                api_context = await self.search_products_from_postgres(conn, id_kenh, standalone_question, res_json.get("filters", {}))
                answer = await (ChatPromptTemplate.from_template(API_RESPONSE_PROMPT) | self.llm | StrOutputParser()).ainvoke({"api_context": api_context, "question": standalone_question})
            else:
                rag_context = await self.search_rag_context(conn, id_kenh, id_ho_so_khach, standalone_question)
                answer = await (ChatPromptTemplate.from_template(SALES_PROMPT) | self.llm | StrOutputParser()).ainvoke({"context": rag_context, "question": standalone_question})

            await conn.execute("INSERT INTO tin_nhan (id_cuoc_hoi_thoai, loai_nguoi_gui, noi_dung, loai_tin_nhan) VALUES ($1, 'bot', $2, 'van_ban')", id_cuoc_hoi_thoai, answer)

rag_service = RAGService()

# --- WORKERS SETUP ---
async def start_chat_worker():
    print("🚀 Worker Chat AI đã kích hoạt...")
    while True:
        try:
            packed = await rag_service.redis.blpop("process_ai_queue", timeout=10)
            if packed:
                _, msg_id_bytes = packed
                asyncio.create_task(rag_service.process_message(msg_id_bytes.decode('utf-8')))
        except Exception:
            await asyncio.sleep(0.2)

async def start_ingest_worker():
    print("🚀 Worker Ingest (Word + Sync Woo) đã kích hoạt...")
    while True:
        try:
            packed = await rag_service.redis.blpop("ingest_queue", timeout=10)
            if packed:
                _, data_bytes = packed
                job = json.loads(data_bytes.decode('utf-8'))
                if job.get("type") == "sync_woo":
                    asyncio.create_task(rag_service.sync_all_woocommerce_data(job["id_kenh"]))
                elif job.get("type") == "word":
                    asyncio.create_task(rag_service.ingest_word_file(
                        file_id=job["file_id"],
                        file_path=job["file_path"],
                        id_ho_so_khach=job["id_ho_so_khach"]
                    ))
        except Exception:
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