import time
import os
import asyncio
import json
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda

# --- CẤU HÌNH SEMAPHORE (Hàng đợi xử lý) ---
MAX_CONCURRENT_REQUESTS = 3
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

vectorstore = None
retriever = None
rag_chain = None
analysis_chain = None
api_response_chain = None
embeddings = None

async def check_concurrency():
    """Dependency kiểm tra số lượng request, nếu đầy sẽ tự đưa vào hàng đợi."""
    async with request_semaphore:
        yield

# Load các biến môi trường
load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CONSUMER_KEY_ENV = os.getenv("CONSUMER_KEY_ENV")
CONSUMER_SECRET_ENV = os.getenv("CONSUMER_SECRET_ENV")


# --- PROMPT 1: PHÂN TÍCH Ý ĐỊNH & VIẾT LẠI CÂU HỎI (GỘP DUY NHẤT) ---
ANALYSIS_PROMPT = """
Bạn là bộ não phân tích ý định khách hàng cho hệ thống Chatbot E-commerce kết nối với WooCommerce.
Nhiệm vụ của bạn là đọc "Lịch sử hội thoại" và "Câu hỏi mới nhất" để thực hiện 2 việc cùng lúc:

1. standalone_question: Viết lại câu hỏi mới nhất thành một câu đầy đủ, rõ ràng, sửa hết đại từ thay thế (nó, cái này, màu đó...) dựa vào lịch sử.
2. target & filters: Phân tích xem câu hỏi này thuộc nhóm nào để hệ thống xử lý:
   - Nếu câu hỏi cần tra cứu dữ liệu thực tế từ kho hàng WooCommerce (Hỏi giá, tìm khoảng giá, check màu sắc, check size, kiểm tra xem còn hàng/tồn kho không, tìm theo danh mục sản phẩm...). Hãy gắn "target": "woocommerce" và trích xuất các bộ lọc tương ứng.
   - Nếu câu hỏi chỉ là hỏi đáp chung (Chính sách đổi trả, bảo hành, địa chỉ shop, tư vấn chất liệu...) hoặc chào hỏi xã giao. Hãy gắn "target": "rag".

ĐẦU RA YÊU CẦU: Chỉ trả về một chuỗi JSON duy nhất, không giải thích gì thêm, tuân thủ cấu trúc sau:

{{
  "standalone_question": "Câu hỏi sau khi đã viết lại đầy đủ ngữ cảnh",
  "target": "woocommerce" hoặc "rag",
  "filters": {{
    "max_price": số_tiền_tối_đa_nếu_khách_yêu_cầu dạng số (hoặc null),
    "min_price": số_tiền_tối_thiểu_nếu_khách_yêu_cầu dạng số (hoặc null),
    "color": "màu sắc khách tìm nếu có" (hoặc null),
    "size": "size khách tìm nếu có" (hoặc null),
    "stock_check": true nếu khách hỏi còn hàng không/còn loại này không (hoặc false),
    "category": "danh mục sản phẩm nếu có, ví dụ: áo khoác, giày..." (hoặc null)
  }}
}}

Lịch sử hội thoại:
{history}

Câu hỏi mới nhất của khách:
{question}
"""

# --- PROMPT 2: RAG CHĂM SÓC KHÁCH HÀNG THÔNG THƯỜNG ---
SALES_PROMPT = """
Bạn là một nhân viên bán hàng chuyên nghiệp.

QUY TẮC CỐT LÕI:
1. Chỉ được trả lời dựa trên thông tin trong tài liệu.
2. Nếu tài liệu có câu trả lời thì cứ trả lời dạ kèm câu trả lời.
3. Nếu tài liệu không chứa câu trả lời thì trả lời chính xác là:'Dạ để em hỏi lại sếp'.
4. Không được suy luận.
5. Không được sử dụng kiến thức bên ngoài.
6. Luôn xưng hô dạ, em. và trả lời một cách ngắn gọn.

Thông tin tài liệu:
{context}

Câu hỏi khách hàng:
{question}
"""

# --- PROMPT 3: TRẢ LỜI CHO KHÁCH DỰA TRÊN KẾT QUẢ WOOCOMMERCE API ---
API_RESPONSE_PROMPT = """
Bạn là một nhân viên bán hàng thân thiện, lễ phép (xưng dạ, em).
Hãy dựa vào danh sách sản phẩm WooCommerce real-time dưới đây để tổng hợp và trả lời câu hỏi của khách một cách hấp dẫn nhưng phải ngắn gọn.
YÊU CẦU:
- Liệt kê mỗi sản phẩm trên một dòng riêng biệt (kết thúc sản phẩm cần 1 dấu xuống dòng).
- Đính kèm link theo định dạng markdown gọn: Đường_link
- Mỗi sản phẩm thêm Ảnh: Đường_link_ảnh
- Nếu danh sách trống, hãy báo lịch sự là hiện tại mẫu này bên em đang hết hàng.

Danh sách sản phẩm từ hệ thống:
{api_context}

Câu hỏi gốc của khách: {question}
"""


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def call_woocommerce_api_advanced(filters: dict):
    """
    Hàm gọi trực tiếp vào website WordPress của khách hàng thông qua WooCommerce REST API.
    Xử lý tìm kiếm thông minh kết hợp lọc thuộc tính động.
    """
    WOO_URL = "https://minhshop.minh2309.io.vn/wp-json/wc/v3/products"
    CONSUMER_KEY = CONSUMER_KEY_ENV
    CONSUMER_SECRET = CONSUMER_SECRET_ENV
    
    # Khởi tạo params mặc định
    params = {
        "status": "publish",
        "per_page": 5
    }
    
    # 1. Bộ lọc số tiền và tồn kho
    if filters.get("max_price"):
        params["max_price"] = filters["max_price"]
    if filters.get("min_price"):
        params["min_price"] = filters["min_price"]
    if filters.get("stock_check") is True:
        params["stock_status"] = "instock"
        
    # 2. Xử lý tên sản phẩm / danh mục (category) bằng lệnh search chính xác
    if filters.get("category"):
        params["search"] = filters["category"]

    # 3. Ghi chú về Màu sắc & Size (Attributes nâng cao)
    # Nếu hệ thống WordPress của bạn cài thêm các plugin bộ lọc như "Premmerce" hoặc "WOOF", 
    # họ sẽ cấp các param dạng ?filter_color=đen hoặc ?filter_size=l. 
    # Nếu không dùng plugin, ta bổ sung từ khóa màu/size thẳng vào lệnh search để bổ trợ tìm kiếm:
    search_keywords = []
    if filters.get("category"):
        search_keywords.append(filters["category"])
    if filters.get("color"):
        search_keywords.append(filters["color"])
    if filters.get("size"):
        search_keywords.append(filters["size"])
        
    if search_keywords:
        params["search"] = " ".join(search_keywords)

    try:
        response = requests.get(WOO_URL, params=params, auth=(CONSUMER_KEY, CONSUMER_SECRET), timeout=5)
        if response.status_code == 200:
            raw_products = response.json()
            simplified_products = []
            if isinstance(raw_products, list):
                for p in raw_products:
                    # Bẫy kiểm tra kỹ hơn bằng Code Python trước khi trả về cho AI:
                    # Nếu AI yêu cầu màu "đỏ", nhưng sản phẩm trả về không chứa từ "đỏ" nào trong dữ liệu, ta có thể lọc bỏ tại đây.
                    simplified_products.append({
                        "name": p.get("name", "Sản phẩm không tên"),
                        "price": p.get("price", "0"),
                        "permalink": p.get("permalink", "#"),
                        "images": p.get("images", [])
                    })
            print(f'woo trả về thành công: {simplified_products}')
            return simplified_products
        else:
            print(f"[API ERROR] Chi tiết lỗi từ WordPress: {response.text}")
        
    except Exception as e:
        print(f"Lỗi kết nối WooCommerce API: {e}")
    return []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore, rag_chain, retriever, analysis_chain, api_response_chain, embeddings

    # 1. Khởi tạo Embeddings
    embeddings = NVIDIAEmbeddings(
        model="nvidia/nv-embed-v1", 
        api_key=NVIDIA_API_KEY
    )

    # 2. Kết nối với ChromaDB
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 2}
    )

    # 3. Khởi tạo LLMs (Sử dụng gemini-3.1-flash-lite để hiểu định dạng JSON tốt và xử lý nhanh)
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
        max_output_tokens=512,
    )
    
    # 4. Thiết lập các Chains hành động độc lập
    
    # Chain 1: Phân tích tích hợp (Gồm viết lại câu hỏi + phân loại ý định)
    analysis_prompt_template = ChatPromptTemplate.from_template(ANALYSIS_PROMPT)
    analysis_chain = analysis_prompt_template | llm | StrOutputParser()
    
    # Chain 2: Xử lý RAG truyền thống
    sales_prompt_template = ChatPromptTemplate.from_template(SALES_PROMPT)
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | sales_prompt_template
        | llm
        | StrOutputParser()
    )
    
    # Chain 3: Xử lý câu trả lời từ API WooCommerce
    api_res_prompt_template = ChatPromptTemplate.from_template(API_RESPONSE_PROMPT)
    api_response_chain = api_res_prompt_template | llm | StrOutputParser()

    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://minhshop.minh2309.io.vn"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ENDPOINT CHAT CHÍNH ---
@app.post("/api/chat", dependencies=[Depends(check_concurrency)])
async def chat(request: dict = Body(...)):
    messages = request.get("messages", [])
    if not messages:
        return {"error": "Không có messages."}

    history_text = "\n".join([
        f"{m['role']}: {m['content']}"
        for m in messages[:-1]
    ])
    latest_question = messages[-1]["content"]

    start_time = time.perf_counter()
    
    # BƯỚC 1: Gọi một lượt duy nhất lên Gemini để Re-phrase & Phân loại cấu trúc
    raw_analysis = analysis_chain.invoke({
        "history": history_text,
        "question": latest_question
    })
    
    # Xử lý làm sạch chuỗi phòng trường hợp Gemini bọc JSON trong tag markdown ```json ... ```
    clean_json = raw_analysis.replace("```json", "").replace("```", "").strip()
    
    try:
        analysis_result = json.loads(clean_json)
    except Exception as e:
        print(f"[ERROR] Không thể parse JSON từ AI. Kết quả thô: {raw_analysis}")
        # Fallback an toàn nếu AI trả về chuỗi lỗi không đúng định dạng JSON
        analysis_result = {
            "standalone_question": latest_question,
            "target": "rag",
            "filters": {}
        }

    standalone_question = analysis_result.get("standalone_question", latest_question)
    target = analysis_result.get("target", "rag")
    filters = analysis_result.get("filters", {})

    # DEBUG LOG TRÊN TERMINAL
    print("\n" + "="*50)
    print(f"LATEST QUESTION: {latest_question}")
    print(f"STANDALONE QUESTION: {standalone_question}")
    print(f"ROUTING TARGET: {target.upper()}")
    print(f"EXTRACTED FILTERS: {filters}")
    print("="*50 + "\n")

    # BƯỚC 2: Rẽ nhánh xử lý dựa trên kết quả phân tích
    if target == "woocommerce":
        # Nhánh 1: Gọi WooCommerce API của WordPress khách hàng
        t_api_start = time.perf_counter()
        products = call_woocommerce_api_advanced(filters)
        
        # Đóng gói danh sách sản phẩm thành ngữ cảnh dạng văn bản (Đã cập nhật để lấy ảnh)
        api_context = ""
        if isinstance(products, list):
            for p in products:
                # Tiến hành lấy URL ảnh đầu tiên trong mảng images của sản phẩm
                img_url = ""
                if p.get("images") and len(p["images"]) > 0:
                    img_url = p["images"][0].get("src", "")
                
                # Ghi nhận thông tin sản phẩm và đính kèm link ảnh ra context thô
                api_context += f"- Tên: {p['name']} | Giá: {p['price']}đ | Link: {p['permalink']}"
                if img_url:
                    api_context += f" | Ảnh: {img_url}"
                api_context += "\n"
        
        # Sinh câu trả lời bán hàng từ dữ liệu API
        answer = api_response_chain.invoke({
            "api_context": api_context if api_context else "Không tìm thấy sản phẩm nào phù hợp.",
            "question": standalone_question
        })
        print(f"[TIMING] Nhánh WooCommerce xử lý xong.")
    else:
        # Nhánh 2: Truy vấn dữ liệu tĩnh (RAG ChromaDB) như cũ
        answer = rag_chain.invoke(standalone_question)
        print(f"[TIMING] Nhánh RAG xử lý xong.")

    t_end = time.perf_counter()
    
    return {
        "answer": answer,
        "timing": {
            "total_seconds": round(t_end - start_time, 4)
        }
    }