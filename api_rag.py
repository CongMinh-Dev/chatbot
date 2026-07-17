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

1. standalone_question: Viết lại câu hỏi mới nhất thành một câu đầy đủ, rõ ràng, sửa hết đại từ thay thế (nó, cái này, màu đó, size đó, dung tích đó...) dựa vào lịch sử.
2. target & filters: Phân tích xem câu hỏi này thuộc nhóm nào để hệ thống xử lý:
   - Nếu câu hỏi cần tra cứu dữ liệu thực tế từ kho hàng WooCommerce. Ví dụ: Hỏi giá, tìm khoảng giá, kiểm tra xem sản phẩm cụ thể còn hàng/tồn kho không, tìm danh sách sản phẩm theo tên/danh mục, hoặc hỏi xem sản phẩm đó có những size/màu sắc/biến thể cụ thể nào để đặt mua. Hãy gắn "target": "woocommerce" và trích xuất các bộ lọc tương ứng.
   - nếu câu hỏi là hỏi đáp chung (Chính sách đổi trả, bảo hành, địa chỉ shop, tư vấn chất liệu tổng quát, chào hỏi xã giao...) HOẶC là câu hỏi nhờ tư vấn chọn size/đo size dựa trên số đo cơ thể. Hãy gắn "target": "rag".

LƯU Ý QUAN TRỌNG CHO TRƯỜNG 'category':
- Vì sản phẩm có thể có vô vàn loại biến thể khác nhau (Màu sắc, Size, Dung tích...). Bạn hãy GOM TẤT CẢ các từ khóa liên quan đến tên loại sản phẩm thực tế khách tìm vào trường "category".
- TUYỆT ĐỐI KHÔNG điền vào trường "category" những từ khóa chung chung, mơ hồ của khách như: "tất cả sản phẩm", "sản phẩm nào", "các sản phẩm", "danh sách sản phẩm", "hàng hóa"... Nếu khách chỉ muốn liệt kê chung chung không có tên sản phẩm cụ thể, hãy để "category": null.
- Nếu khách hỏi: "thời trang nữ", "đồ con gái"... -> LUÔN ĐỂ "category": "Đồ Nữ"
- Nếu khách hỏi: "đồ nam", "thời trang nam"... -> LUÔN ĐỂ "category": "Đồ Nam"

ĐẦU RA YÊU CẦU: Chỉ trả về một chuỗi JSON duy nhất, không giải thích gì thêm, tuân thủ cấu trúc sau:

{{
  "standalone_question": "Câu hỏi sau khi đã viết lại đầy đủ ngữ cảnh",
  "target": "woocommerce" hoặc "rag",
  "filters": {{
    "max_price": số_tiền_tối_đa_nếu_khách_yêu_cầu dạng số (hoặc null),
    "min_price": số_tiền_tối_thiểu_nếu_khách_yêu_cầu dạng số (hoặc null),
    "stock_check": true nếu khách hỏi còn hàng không/còn loại này không (hoặc false),
    "category": Tên sản phẩm cụ thể (hoặc null),
    "get_total_count": true nếu khách hỏi "bao nhiêu sản phẩm" (hỏi số lượng). Gán false nếu khách hỏi "sản phẩm nào" (hỏi danh sách liệt kê cụ thể),
    "page": số_trang_cần_lấy_dữ_liệu (Mặc định là 1. Nếu khách có ý định muốn xem các mẫu còn lại, xem tiếp các sản phẩm khác, hoặc xem trang sau dựa vào lịch sử hội thoại, hãy tăng số này lên thành 2, hoặc 3...)
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
Bạn là một nhân viên bán hàng chuyên nghiệp, thân thiện và lễ phép (luôn xưng dạ, em).
Hãy dựa vào danh sách sản phẩm WooCommerce real-time dưới đây để trả lời câu hỏi của khách một cách CỰC KỲ NGẮN GỌN và TRỰC TIẾP.

QUY TẮC CỐT LÕI (KHÔNG ĐƯỢC QUÊN):
1. PHÂN BIỆT Ý ĐỊNH HỎI SỐ LƯỢNG VÀ YÊU CẦU XEM SẢN PHẨM:
   - TRƯỜNG HỢP A (Hỏi tổng số lượng sản phẩm): bạn CHỈ cần trả lời trực tiếp số lượng dựa trên "total_count_info" trong dữ liệu hệ thống.
   - TRƯỜNG HỢP B (Xem sản phẩm hoặc Xem tiếp sản phẩm):
     + Nếu trong "[THÔNG TIN HỆ THỐNG]" báo đang ở trang số 1 VÀ khách muốn xem danh sách sản phẩm cụ thể, nhưng số lượng sản phẩm trả về ít hơn số lượng khách yêu cầu trong câu hỏi, hãy trả lời chính xác theo cấu trúc sau: "em gửi trước các sản phẩm này nha, các sản phẩm còn lại thì lên web tham khảo giúp em: danh sách sản phẩm từ hệ thống". không được kèm câu: Dạ, hiện tại trên website bên em đang có....
     + Nếu trong "[THÔNG TIN HỆ THỐNG]" báo đang ở trang số 2 trở đi: Bạn PHẢI hiểu đây là danh sách các sản phẩm tiếp theo (còn lại) trong kho và liệt kê chúng ra một cách tự nhiên.

2. RÀNG BUỘC BIẾN THỂ THỰC TẾ (TUYỆT ĐỐI KHÔNG SUY DIỄN):
   - Tuyệt đối không phỏng đoán logic để tự bịa ra bất kỳ thông số, kích thước, màu sắc hay phiên bản nào khác nếu dữ liệu hệ thống không liệt kê. Nếu hệ thống báo hết hoặc không ghi, nghĩa là HẾT HÀNG.

3. CHIẾN LƯỢC TƯ VẤN KHI HẾT HÀNG (UPSELL/CROSS-SELL):
   - TRƯỜNG HỢP 1 (Hết một vài biến thể): Nếu khách hỏi trúng một lựa chọn/biến thể đã hết, nhưng sản phẩm đó VẪN CÒN các biến thể khác trong dữ liệu -> Hãy báo hết và chủ động gợi ý khách chuyển sang các lựa chọn/biến thể còn lại (chỉ đích danh các biến thể thực tế đang còn trong data).
   - TRƯỜNG HỢP 2 (Hết sạch toàn bộ sản phẩm hoặc Không tìm thấy): Nếu trong dữ liệu có xuất hiện danh sách "Sản phẩm gợi ý thay thế" -> Hãy báo lịch sự là mẫu khách tìm đang hết hàng, và ngay lập tức giới thiệu các sản phẩm thay thế được cung cấp này (nêu rõ chúng có tính năng, công dụng tương tự hoặc cùng phân khúc).

4. NGUYÊN LIỆU BẮT BUỘC: Dù tư vấn thế nào, đối với các sản phẩm còn hàng hoặc sản phẩm gợi ý, Nếu sản phẩm có ảnh thì phải có link ảnh (gửi 1 ảnh thôi, khi nào khách yêu cầu gửi nhiều ảnh thì mới gửi nhiều ảnh)

5. CHỐNG BỊA ĐẶT DỮ LIỆU (QUAN TRỌNG):
   - Tuyệt đối không tự ý sinh ra các sản phẩm ảo dạng "[Tên sản phẩm 1]", "[Link ảnh 1]" hoặc tự bịa ra các link ảnh không có thật trong dữ liệu hệ thống.
   - Nếu dữ liệu báo gặp lỗi kết nối hệ thống hoặc trống rỗng, hãy xin lỗi khách một cách lịch sự và báo rằng hệ thống tra cứu đang bận hoặc gặp sự cố kết nối, hẹn khách kiểm tra lại sau ít phút.

Danh sách sản phẩm từ hệ thống:
{api_context}

Câu hỏi gốc của khách: {question}
"""


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_category_id_by_slug(slug_name: str) -> int:
    """
    Hàm phụ trợ lấy ID danh mục dựa trên slug danh mục từ WooCommerce API.
    """
    WOO_CAT_URL = "https://minhshop.minh2309.io.vn/wp-json/wc/v3/products/categories"
    CONSUMER_KEY = CONSUMER_KEY_ENV
    CONSUMER_SECRET = CONSUMER_SECRET_ENV
    
    try:
        response = requests.get(WOO_CAT_URL, auth=(CONSUMER_KEY, CONSUMER_SECRET), params={"slug": slug_name}, timeout=5)
        if response.status_code == 200:
            cats = response.json()
            if cats and isinstance(cats, list) and len(cats) > 0:
                return cats[0].get("id")
    except Exception as e:
        print(f"[ERROR] Lỗi khi lấy ID danh mục cho slug '{slug_name}': {e}")
    return None

def call_woocommerce_api_advanced(filters: dict):
    """
    Hàm Ultimate: Xử lý mượt mà cho cả sản phẩm ĐƠN GIẢN và sản phẩm BIẾN THỂ.
    Tự động gom nhóm động trạng thái hết hàng và trích xuất ẢNH của từng biến thể.
    """
    WOO_URL = "https://minhshop.minh2309.io.vn/wp-json/wc/v3/products"
    CONSUMER_KEY = CONSUMER_KEY_ENV
    CONSUMER_SECRET = CONSUMER_SECRET_ENV
    
    params = {
        "status": "publish",
        "per_page": 5,
        "page": filters.get("page", 1)
    }
    
    if filters.get("max_price"): params["max_price"] = filters["max_price"]
    if filters.get("min_price"): params["min_price"] = filters["min_price"]
    if filters.get("stock_check") is True: params["stock_status"] = "instock"

    category_search = filters.get("category")
    invalid_keywords = ["tất cả sản phẩm", "các sản phẩm", "sản phẩm nào", "danh sách sản phẩm"]

    # Bản đồ ánh xạ từ Tên danh mục viết thường sang Slug tương ứng trong Flatsome
    category_slug_map = {
        "đồ nữ": "do-nu",
        "đồ nam": "do-nam",
        "quần": "quan",
        "áo": "ao",
        "tất cả sản phẩm": "tat-ca-san-pham"
    }

    if category_search:
        category_clean = category_search.lower().strip()
        if category_clean not in invalid_keywords:
            # Nếu phát hiện từ khóa khớp danh mục hệ thống
            if category_clean in category_slug_map:
                target_slug = category_slug_map[category_clean]
                cat_id = get_category_id_by_slug(target_slug)
                if cat_id:
                    params["category"] = cat_id  # Lọc chuẩn theo ID danh mục!
                else:
                    # Dự phòng nếu không lấy được ID từ API danh mục
                    params["search"] = category_search
            else:
                # Nếu không thuộc danh mục tĩnh, xem như từ khóa tìm sản phẩm cụ thể (ví dụ: "áo thun cổ V")
                params["search"] = category_search

    try:
        response = requests.get(WOO_URL, params=params, auth=(CONSUMER_KEY, CONSUMER_SECRET), timeout=5)
        if response.status_code == 200:
            total_products = response.headers.get("X-WP-Total")
            if filters.get("get_total_count") is True:
                return [{
                    "name": "Hệ thống cửa hàng",
                    "total_count_info": f"Tổng số sản phẩm đang có trên website là {total_products} sản phẩm.",
                    "price": "0",
                    "permalink": "#"
                }]

            raw_products = response.json()
            simplified_products = []
            
            if isinstance(raw_products, list):
                for p in raw_products:
                    product_id = p.get("id")
                    product_type = p.get("type", "simple")
                    
                    # 1. LẤY THUỘC TÍNH TỔNG QUÁT
                    variant_list = []
                    if p.get("attributes"):
                        for attr in p["attributes"]:
                            if attr.get("variation") is True or attr.get("visible") is True:
                                name = attr.get("name", "")
                                options = attr.get("options", [])
                                if options:
                                    variant_list.append(f"{name}: {', '.join(options)}")
                    variants_string = " | ".join(variant_list) if variant_list else "Tiêu chuẩn"

                    # 2. XỬ LÝ TRẠNG THÁI HẾT HÀNG & ẢNH CHI TIẾT TỪNG BIẾN THỂ
                    outofstock_string = ""
                    variant_images = []
                    
                    # --- NHÁNH A: SẢN PHẨM BIẾN THỂ (VARIABLE) ---
                    if product_type == "variable":
                        var_url = f"{WOO_URL}/{product_id}/variations"
                        try:
                            # Nâng timeout lên 6s để tránh ngắt kết nối mạng chập chờn
                            var_response = requests.get(var_url, auth=(CONSUMER_KEY, CONSUMER_SECRET), timeout=6)
                            
                            if var_response.status_code == 200:
                                variations_data = var_response.json()
                                generic_groups = {}
                                outofstock_variants = []
                                
                                for v in variations_data:
                                    attrs = v.get("attributes", [])
                                    attr_values = [a.get("option", "") for a in attrs if a.get("option")]
                                    variant_name = " ".join(attr_values)
                                    
                                    if v.get("image") and v["image"].get("src"):
                                        variant_images.append({
                                            "label": variant_name,
                                            "src": v["image"]["src"]
                                        })
                                    
                                    is_out = (v.get("stock_status") == "outofstock" or 
                                              v.get("stock_quantity") == 0 or 
                                              v.get("is_in_stock") is False)
                                    
                                    if is_out:
                                        if len(attr_values) >= 2:
                                            main_key = " ".join(attr_values[:-1])
                                            last_val = attr_values[-1]
                                            
                                            if main_key not in generic_groups:
                                                generic_groups[main_key] = []
                                            generic_groups[main_key].append(last_val)
                                        elif len(attr_values) == 1:
                                            standalone_val = attr_values[0]
                                            if standalone_val not in outofstock_variants:
                                                outofstock_variants.append(f"{standalone_val} hết hàng")

                                for main_key, sub_vals in generic_groups.items():
                                    outofstock_variants.append(f"{main_key} {', '.join(sub_vals)} hết hàng")
                                
                                if outofstock_variants:
                                    outofstock_string = f"({', '.join(outofstock_variants)})"
                            else:
                                outofstock_string = "(Tạm thời không xác định được trạng thái kho)"
                        except Exception as inner_e:
                            # Nếu lỗi kết nối API con, đánh dấu để không xử lý rác
                            print(f"[ERROR] Lỗi lấy biến thể của sản phẩm {product_id}: {inner_e}")
                            outofstock_string = "(Lỗi kết nối kho hàng biến thể)"

                    # --- NHÁNH B: SẢN PHẨM ĐƠN GIẢN (SIMPLE) ---
                    elif p.get("stock_status") == "outofstock" or p.get("stock_quantity") == 0:
                        outofstock_string = "(Hết hàng)"

                    simplified_products.append({
                        "name": p.get("name", "Sản phẩm không tên"),
                        "price": p.get("price", "0"),
                        "permalink": p.get("permalink", "#"),
                        "images": p.get("images", []),
                        "variant_images": variant_images,
                        "variants": variants_string,
                        "outofstock_info": outofstock_string  
                    })
            
            print(f'woo trả về thành công: {simplified_products}')
            return simplified_products
        else:
            print(f"[API ERROR] Chi tiết lỗi từ WordPress: {response.text}")
        
    except Exception as e:
        print(f"Lỗi kết nối WooCommerce API: {e}")
    # Trả về None thay vì [] khi thực sự lỗi kết nối để phân biệt với "Hết hàng/Hết trang"
    return None


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

    # 3. Khởi tạo LLMs
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
        max_output_tokens=512,
    )
    llm0 = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.0,
        max_output_tokens=512,
    )
    
    # Chain 1: Phân tích tích hợp
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
        | llm0
        | StrOutputParser()
    )
    
    # Chain 3: Xử lý câu trả lời từ API WooCommerce
    api_res_prompt_template = ChatPromptTemplate.from_template(API_RESPONSE_PROMPT)
    api_response_chain = api_res_prompt_template | llm0 | StrOutputParser()

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
    
    # BƯỚC 1: Gọi phân tích Re-phrase
    raw_analysis = analysis_chain.invoke({
        "history": history_text,
        "question": latest_question
    })
    
    clean_json = raw_analysis.replace("```json", "").replace("```", "").strip()
    
    try:
        analysis_result = json.loads(clean_json)
    except Exception as e:
        print(f"[ERROR] Không thể parse JSON từ AI. Kết quả thô: {raw_analysis}")
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
        # 1. Gọi WooCommerce API lấy sản phẩm
        products = call_woocommerce_api_advanced(filters)
        current_page = filters.get("page", 1)
        
        # Kiểm tra xem API có thực sự lỗi kết nối (None) hay không
        if products is None:
            api_context = "[HỆ THỐNG GẶP LỖI KẾT NỐI MẠNG ĐẾN WOOCOMMERCE. HÃY BÁO LỖI LỊCH SỰ VỚI KHÁCH HÀNG]"
        else:
            # 2. Xử lý nới lỏng bộ lọc hoặc nhận diện hết trang
            suggested_products = []
            is_all_out = True
            
            if products:
                for p in products:
                    if p.get("outofstock_info") != "(Hết hàng)":
                        is_all_out = False
                        break
            
            # CHỈ TÌM SẢN PHẨM THAY THẾ Ở TRANG 1 (nếu trang 1 hết hàng hoặc không có sản phẩm)
            if current_page == 1 and (not products or is_all_out):
                original_category = filters.get("category", "")
                broad_keyword = original_category.split()[0] if original_category else ""
                if broad_keyword:
                    fallback_filters = {
                        "category": broad_keyword,
                        "stock_check": True,
                        "max_price": filters.get("max_price"),
                        "min_price": filters.get("min_price")
                    }
                    suggested_products = call_woocommerce_api_advanced(fallback_filters)

            # 3. ĐỒNG BỘ ẢNH BIẾN THỂ VÀO NGỮ CẢNH VĂN BẢN
            api_context = ""
            api_context += f"[THÔNG TIN HỆ THỐNG: Đang hiển thị dữ liệu ở trang số {current_page}]\n\n"
            
            # Xử lý trường hợp HẾT TRANG (vượt quá số sản phẩm hiện có khi page > 1)
            if not products and current_page > 1:
                api_context += f"[THÔNG TIN HỆ THỐNG: Đã hết sản phẩm ở các trang tiếp theo. Không còn sản phẩm nào khác ở trang {current_page}. Hãy báo cho khách biết một cách lịch sự là đã xem hết sản phẩm rồi và gợi ý họ quay lại trang trước hoặc tìm từ khóa khác].\n"
            
            elif products:
                api_context += "--- CÁC SẢN PHẨM KHÁCH ĐANG TÌM KIẾM: ---\n"
                for p in products:
                    if p.get("total_count_info"):
                        api_context += f"- Thông tin hệ thống: {p['total_count_info']}\n"
                        continue
                    api_context += f"- Tên: {p['name']} | Biến thể hiện có: {p.get('variants', 'Tiêu chuẩn')}"
                    if p.get("outofstock_info"):
                        api_context += f" | Thông tin hết hàng: {p['outofstock_info']}"
                    api_context += f" | Giá: {p['price']}đ | Link: {p['permalink']}"
                    
                    # ĐƯA DANH SÁCH ẢNH CỦA TỪNG BIẾN THỂ VÀO CHO LLM
                    if p.get("variant_images"):
                        img_details = [f"Ảnh của bản {img['label']}: {img['src']}" for img in p["variant_images"]]
                        api_context += f" | Danh sách ảnh biến thể: [{', '.join(img_details)}]"
                    
                    # Fallback nếu không có ảnh biến thể thì đưa mảng ảnh của sản phẩm tổng
                    elif p.get("images"):
                        img_url = p["images"][0]["src"]
                        api_context += f" | Ảnh: {img_url}"
                        
                    api_context += "\n"
            
            # Nạp sản phẩm gợi ý thay thế (chỉ hiển thị ở trang 1)
            if suggested_products and current_page == 1:
                api_context += "\n--- DANH SÁCH SẢN PHẨM GỢI Ý THAY THẾ (VÌ SẢN PHẨM TRÊN HẾT HÀNG): ---\n"
                for p in suggested_products:
                    # Tránh bị lỗi NoneType nếu fallback_filters cũng bị lỗi mạng trả về None
                    if p is None: 
                        continue
                    if products and p['name'] in [prod['name'] for prod in products if prod is not None]:
                        continue
                    img_url = p["images"][0]["src"] if p.get("images") else ""
                    api_context += f"- Tên sản phẩm thay thế: {p['name']} | Giá: {p['price']}đ | Link: {p['permalink']}"
                    if img_url: api_context += f" | Ảnh: {img_url}"
                    api_context += "\n"

            if not api_context.strip() or api_context.strip() == f"[THÔNG TIN HỆ THỐNG: Đang hiển thị dữ liệu ở trang số {current_page}]":
                api_context = "Hiện tại hệ thống không tìm thấy sản phẩm nào phù hợp và cũng không có sản phẩm thay thế."

        # Sinh câu trả lời từ LLM
        answer = api_response_chain.invoke({
            "api_context": api_context,
            "question": standalone_question
        })
        print(f"[TIMING] Nhánh WooCommerce xử lý xong.")
        
    else:
        # Nhánh 2: Truy vấn dữ liệu tĩnh (RAG ChromaDB) như cũ
        answer = rag_chain.invoke(standalone_question)
        print(f"[TIMING] Nhánh RAG xử lý xong.")

    t_end = time.perf_counter()
    total_seconds = round(t_end - start_time, 4)
    print(f"bot trả lời: {answer} \n thời gian: {total_seconds}")
    return {
        "answer": answer,
        "timing": {
            "total_seconds": total_seconds
        }
    }