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

vectorstore = None
rag_chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore, rag_chain

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

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
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

    # 4. Tạo chain RAG
    template = (
        SALES_PROMPT
        + "\n\nContext:\n{context}\n\nQuestion:\n{question}"
    )
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {
            "context": lambda x: retriever.invoke(x), # Ép dùng nguyên văn
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
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
async def chat(request: dict = Body(..., example={"message": "Sản phẩm nào giúp tăng sức đề kháng?"})):
    user_message = request.get("message")
    if not user_message:
        return {"error": "Vui lòng cung cấp nội dung tin nhắn."}

    # Đo thời gian từng bước
    start_time = time.perf_counter()
    
    # 1. Bước lấy context (Retrieval)
    t0 = time.perf_counter()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    context = retriever.invoke(user_message)
    t1 = time.perf_counter()
    
    # 2. Bước tạo phản hồi (Generation)
    response = rag_chain.invoke(user_message)
    print(f"Model response: {response}")
    t2 = time.perf_counter()

    return {
        "answer": response,
        "timing": {
            "retrieval_seconds": round(t1 - t0, 4),
            "generation_seconds": round(t2 - t1, 4),
            "total_seconds": round(t2 - start_time, 4)
        }
    }