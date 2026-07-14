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
   - Nếu câu hỏi cần tra cứu dữ liệu thực tế từ kho hàng WooCommerce (Hỏi giá, tìm khoảng giá, kiểm tra xem còn hàng/tồn kho không, tìm theo tên sản phẩm hoặc bất kỳ thuộc tính biến thể nào như màu sắc, kích cỡ, dung tích, phiên bản, các câu hỏi thống kê tổng số lượng sản phẩm, tổng số danh mục hiện có...). Hãy gắn "target": "woocommerce" và trích xuất các bộ lọc tương ứng.
   - Nếu câu hỏi chỉ là hỏi đáp chung (Chính sách đổi trả, bảo hành, địa chỉ shop, tư vấn chất liệu tổng quát...) hoặc chào hỏi xã giao. Hãy gắn "target": "rag".

LƯU Ý QUAN TRỌNG CHO TRƯỜNG 'category':
- Vì sản phẩm có thể có vô vàn loại biến thể khác nhau (Màu sắc, Size, Dung tích...). Bạn hãy GOM TẤT CẢ các từ khóa liên quan đến tên loại sản phẩm thực tế khách tìm vào trường "category".
- TUYỆT ĐỐI KHÔNG điền vào trường "category" những từ khóa chung chung, mơ hồ của khách như: "tất cả sản phẩm", "sản phẩm nào", "các sản phẩm", "danh sách sản phẩm", "hàng hóa"... Nếu khách chỉ muốn liệt kê chung chung không có tên sản phẩm cụ thể, hãy để "category": null.

ĐẦU RA YÊU CẦU: Chỉ trả về một chuỗi JSON duy nhất, không giải thích gì thêm, tuân thủ cấu trúc sau:

{{
  "standalone_question": "Câu hỏi sau khi đã viết lại đầy đủ ngữ cảnh",
  "target": "woocommerce" hoặc "rag",
  "filters": {{
    "max_price": số_tiền_tối_đa_nếu_khách_yêu_cầu dạng số (hoặc null),
    "min_price": số_tiền_tối_thiểu_nếu_khách_yêu_cầu dạng số (hoặc null),
    "stock_check": true nếu khách hỏi còn hàng không/còn loại này không (hoặc false),
    "category": Tên sản phẩm cụ thể (hoặc null),
    "get_total_count": true nếu khách hỏi "bao nhiêu sản phẩm" (hỏi số lượng). Gán false nếu khách hỏi "sản phẩm nào" (hỏi danh sách liệt kê cụ thể)
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
1. KHÔNG DÀI DÒNG, KHÔNG THỪA THÃI: 
   - Chỉ trả lời chính xác, trực tiếp vào câu hỏi của khách dựa trên dữ liệu được cung cấp.
   - TUYỆT ĐỐI không tự ý thêm các câu chào hỏi xã giao, câu kết thúc rườm rà (ví dụ: "Nếu anh/chị cần em hỗ trợ thêm...", "Em cảm ơn anh/chị...").

2. XỬ LÝ KHI SỐ LƯỢNG HIỂN THỊ ÍT HƠN THỰC TẾ (QUAN TRỌNG):
   - Nếu trong câu hỏi của khách hoặc lịch sử có nhắc đến một con số cụ thể (ví dụ: "8 sản phẩm nào thế"), nhưng danh sách hệ thống trả về hiện tại ít hơn con số đó (ví dụ chỉ có 5 sản phẩm).
   - Bạn PHẢI trả lời khéo léo theo hướng: Liệt kê trước các sản phẩm nổi bật/mới nhất, và chủ động mời khách bấm vào link website hoặc nhắn cụ thể để xem nốt các mẫu còn lại.
   - Ví dụ mẫu: "Dạ, em xin phép gợi ý trước 5 sản phẩm nổi bật nhất trong số 8 sản phẩm của cửa hàng mình ạ:..." hoặc "Dạ, đây là 5 mẫu đang sẵn hàng/mới nhất trong số 8 sản phẩm ạ:..."

3. RÀNG BUỘC BIẾN THỂ THỰC TẾ (TUYỆT ĐỐI KHÔNG SUY DIỄN):
   - Tuyệt đối không phỏng đoán logic để tự bịa ra bất kỳ thông số, kích thước, màu sắc hay phiên bản nào khác nếu dữ liệu hệ thống không liệt kê. Nếu hệ thống báo hết hoặc không ghi, nghĩa là HẾT HÀNG.

4. CHIẾN LƯỢC TƯ VẤN KHI HẾT HÀNG (UPSELL/CROSS-SELL):
   - TRƯỜNG HỢP 1 (Hết một vài biến thể): Nếu khách hỏi trúng một lựa chọn/biến thể đã hết, nhưng sản phẩm đó VẪN CÒN các biến thể khác trong dữ liệu -> Hãy báo hết và chủ động gợi ý khách chuyển sang các lựa chọn/biến thể còn lại (chỉ đích danh các biến thể thực tế đang còn trong data).
   - TRƯỜNG HỢP 2 (Hết sạch toàn bộ sản phẩm hoặc Không tìm thấy): Nếu trong dữ liệu có xuất hiện danh sách "Sản phẩm gợi ý thay thế" -> Hãy báo lịch sự là mẫu khách tìm đang hết hàng, và ngay lập tức giới thiệu các sản phẩm thay thế được cung cấp này (nêu rõ chúng có tính năng, công dụng tương tự hoặc cùng phân khúc).

5. NGUYÊN LIỆU BẮT BUỘC: Dù tư vấn thế nào, đối với các sản phẩm còn hàng hoặc sản phẩm gợi ý, phải có link ảnh (Nếu sản phẩm có ảnh).

Danh sách sản phẩm từ hệ thống:
{api_context}

Câu hỏi gốc của khách: {question}
"""


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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
        "per_page": 5
    }
    
    if filters.get("max_price"): params["max_price"] = filters["max_price"]
    if filters.get("min_price"): params["min_price"] = filters["min_price"]
    if filters.get("stock_check") is True: params["stock_status"] = "instock"
    category_search = filters.get("category")
    invalid_keywords = ["tất cả sản phẩm", "các sản phẩm", "sản phẩm nào", "danh sách sản phẩm"]
    if category_search:
        # Nếu từ khóa nằm trong danh sách rác, bỏ qua không truyền vào params["search"]
        if category_search.lower().strip() in invalid_keywords:
            pass 
        else:
            params["search"] = category_search

    try:
        response = requests.get(WOO_URL, params=params, auth=(CONSUMER_KEY, CONSUMER_SECRET), timeout=5)
        if response.status_code == 200:
            # đoạn xử lý tổng số lượng sản phẩm
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
                    variant_images = []  # Danh sách chứa ảnh cụ thể của từng biến thể nếu có
                    
                    # --- NHÁNH A: SẢN PHẨM BIẾN THỂ (VARIABLE) ---
                    if product_type == "variable":
                        var_url = f"{WOO_URL}/{product_id}/variations"
                        var_response = requests.get(var_url, auth=(CONSUMER_KEY, CONSUMER_SECRET), timeout=4)
                        
                        if var_response.status_code == 200:
                            variations_data = var_response.json()
                            generic_groups = {}
                            outofstock_variants = []
                            
                            for v in variations_data:
                                # Trích xuất thông tin thuộc tính của biến thể này (ví dụ: Màu: Đen, Size: L)
                                attrs = v.get("attributes", [])
                                attr_values = [a.get("option", "") for a in attrs if a.get("option")]
                                variant_name = " ".join(attr_values)
                                
                                # ĐỒNG BỘ ẢNH BIẾN THỂ: Nếu biến thể này có cấu hình ảnh riêng, lưu lại kèm nhãn tên
                                if v.get("image") and v["image"].get("src"):
                                    variant_images.append({
                                        "label": variant_name,
                                        "src": v["image"]["src"]
                                    })
                                
                                # Kiểm tra tồn kho biến thể
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

                    # --- NHÁNH B: SẢN PHẨM ĐƠN GIẢN (SIMPLE) ---
                    elif p.get("stock_status") == "outofstock" or p.get("stock_quantity") == 0:
                        outofstock_string = "(Hết hàng)"

                    # Đóng gói kết quả đồng nhất trả về
                    simplified_products.append({
                        "name": p.get("name", "Sản phẩm không tên"),
                        "price": p.get("price", "0"),
                        "permalink": p.get("permalink", "#"),
                        "images": p.get("images", []),          # Giữ ảnh gốc của sản phẩm cha
                        "variant_images": variant_images,      # NÂNG CẤP: Gửi kèm ảnh định danh của từng biến thể
                        "variants": variants_string,
                        "outofstock_info": outofstock_string  
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
        # 1. Gọi WooCommerce API lấy sản phẩm
        products = call_woocommerce_api_advanced(filters)
        
        # 2. Xử lý nới lỏng bộ lọc tìm sản phẩm thay thế nếu hết hàng (giữ nguyên logic cũ của bạn)
        suggested_products = []
        is_all_out = True
        if products:
            for p in products:
                if p.get("outofstock_info") != "(Hết hàng)":
                    is_all_out = False
                    break
        if not products or is_all_out:
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

        # 3. ĐỒNG BỘ ẢNH BIẾN THỂ VÀO NGỮ CẢNH VĂN BẢN (NÂNG CẤP Ở ĐÂY)
        api_context = ""
        
        if products:
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
        
        # Nạp sản phẩm gợi ý thay thế
        if suggested_products:
            api_context += "\n--- DANH SÁCH SẢN PHẨM GỢI Ý THAY THẾ (VÌ SẢN PHẨM TRÊN HẾT HÀNG): ---\n"
            for p in suggested_products:
                if products and p['name'] in [prod['name'] for prod in products]:
                    continue
                img_url = p["images"][0]["src"] if p.get("images") else ""
                api_context += f"- Tên sản phẩm thay thế: {p['name']} | Giá: {p['price']}đ | Link: {p['permalink']}"
                if img_url: api_context += f" | Ảnh: {img_url}"
                api_context += "\n"

        if not api_context:
            api_context = "Hiện tại hệ thống không tìm thấy sản phẩm nào phù hợp và cũng không có sản phẩm thay thế."

        # Sinh câu trả lời
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
    
    return {
        "answer": answer,
        "timing": {
            "total_seconds": round(t_end - start_time, 4)
        }
    }
