import time
import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- CẤU HÌNH SEMAPHORE (Hàng đợi xử lý) ---
# Chỉ cho phép tối đa 3 request/1 worker(lõi cpu) xử lý đồng thời, các request khác sẽ đợi
MAX_CONCURRENT_REQUESTS = 3
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

vectorstore = None
llm = None

async def check_concurrency():
    """Dependency kiểm tra số lượng request, nếu đầy sẽ tự đưa vào hàng đợi."""
    async with request_semaphore:
        yield

# Load các biến môi trường
load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

SALES_PROMPT = (
    "Bạn là một nhân viên bán hàng chuyên nghiệp, chỉ cung cấp thông tin có sẵn trong tài liệu.\n"
    "QUY TẮC CỐT LÕI:\n"
    "1) Chỉ trả lời DỰA TRÊN THÔNG TIN TÌM THẤY trong ngữ cảnh cung cấp.\n"
    "2) Nếu thông tin không chứa chính xác tiêu chí khách hàng hỏi, "
    "hãy trả lời đúng nguyên văn: 'Dạ để em hỏi lại sếp'.\n"
    "3) Không được tự ý liên kết các khái niệm, không tự suy luận, không tự đặt câu hỏi lại cho khách hàng.\n"
    "4) Luôn xưng hô 'dạ', 'em' với khách hàng.\n"
)
REWRITE_PROMPT = """
Dựa vào lịch sử hội thoại, hãy viết lại câu hỏi cuối cùng
thành một câu hỏi độc lập, đầy đủ ngữ cảnh.

Chỉ trả về câu hỏi đã viết lại.
Không giải thích.

Lịch sử:

{history}

Câu hỏi cuối:

{question}
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore, llm

    # 1. Sử dụng OllamaEmbeddings (Đảm bảo base_url đúng IP LXC của bạn)
    embeddings = NVIDIAEmbeddings(
    model="nvidia/nv-embed-v1", # Hoặc model phù hợp khác
    api_key=NVIDIA_API_KEY
    )

    # 2. Kết nối với ChromaDB
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    

    # 3. Khởi tạo LLM NVIDIA (Gemma-4)
    llm = ChatOpenAI(
        model="google/gemma-3n-e4b-it", # Đổi model
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY,
        temperature=0.0, # Theo cấu hình yêu cầu
        model_kwargs={
            "extra_body": {
                "max_tokens": 512, # Theo cấu hình yêu cầu
                "top_p": 0.70,     # Theo cấu hình yêu cầu
                "frequency_penalty": 0.00,
                "presence_penalty": 0.00
            }
        }
    )

    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://minhshop.minh2309.io.vn"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ENDPOINT CHAT (Áp dụng cơ chế hàng đợi) ---
@app.post("/api/chat", dependencies=[Depends(check_concurrency)])
async def chat(request: dict = Body(...)):

    messages = request.get("messages", [])

    if not messages:
        return {
            "error": "Không có messages."
        }

    history_text = "\n".join([
    f"{m['role']}: {m['content']}"
    for m in messages[:-1]
    ])

    latest_question = messages[-1]["content"]

    

    # =========================
    # RETRIEVER
    # =========================

    start_time = time.perf_counter()

    t0 = time.perf_counter()

    rewrite_messages = [
    {
        "role": "system",
        "content": REWRITE_PROMPT.format(
            history=history_text,
            question=latest_question
        )
    }
    ]

    rewrite_response = llm.invoke(
        rewrite_messages
    )

    standalone_question = rewrite_response.content.strip()

    print("Standalone Question:")
    print(standalone_question)

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    docs = retriever.invoke(standalone_question)

    t1 = time.perf_counter()

    # =========================
    # BUILD CONTEXT
    # =========================

    context_text = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # =========================
    # INJECT CONTEXT
    # =========================

    request_messages = messages.copy()

    request_messages[-1]["content"] = f"""
    Thông tin tài liệu:

    {context_text}

    Lịch sử hội thoại:

    {history_text}

    Câu hỏi khách hàng:

    {latest_question}
    """


    # =========================
    # SYSTEM PROMPT
    # =========================

    final_messages = [
    {
        "role": "system",
        "content": SALES_PROMPT
    }
    ] + request_messages

    # =========================
    # GENERATE
    # =========================

    response = llm.invoke(final_messages)

    answer = response.content

    t2 = time.perf_counter()

    print(f"Model response: {answer}")

    return {
        "answer": answer,
        "timing": {
            "retrieval_seconds": round(t1 - t0, 4),
            "generation_seconds": round(t2 - t1, 4),
            "total_seconds": round(t2 - start_time, 4)
        }
    }
