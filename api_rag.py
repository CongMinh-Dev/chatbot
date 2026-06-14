import time
import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, Body, Depends
from contextlib import asynccontextmanager

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
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
    "Bạn là một nhân viên bán hàng chuyên nghiệp, luôn lịch sự, niềm nở và xưng hô 'dạ', 'em' với khách hàng.\n"
    "QUY TẮC CỐT LÕI:\n"
    "1) Chỉ trả lời dựa trên thông tin có trong tài liệu.\n"
    "2) Khi khách hàng hỏi về danh sách sản phẩm theo tiêu chí (giá, công dụng, xuất xứ...), "
    "hãy liệt kê ĐẦY ĐỦ tất cả các sản phẩm tìm thấy trong tài liệu thỏa mãn điều kiện đó.\n"
    "3) Nếu không tìm thấy thông tin, trả lời đúng nguyên văn: 'Dạ để em hỏi lại sếp'.\n"
    "4) Không tự suy luận ngoài tài liệu.\n"
)

vectorstore = None
rag_chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore, rag_chain

    # 1. Sử dụng OllamaEmbeddings (Đảm bảo base_url đúng IP LXC của bạn)
    embeddings = OllamaEmbeddings(
        model="bge-m3:latest", 
        base_url="http://192.168.1.100:11434" # đây là IP LXC thực tế của tôi
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
        model="google/gemma-4-31b-it",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY,
        temperature=0.0,
        model_kwargs={
            "extra_body": {
                "max_tokens": 16384,
                "top_p": 0.95,
                "chat_template_kwargs": {"enable_thinking": True}
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
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    yield

app = FastAPI(lifespan=lifespan)

# --- ENDPOINT CHAT (Áp dụng cơ chế hàng đợi) ---
@app.post("/api/chat", dependencies=[Depends(check_concurrency)])
async def chat(request: dict = Body(...)):
    user_message = request.get("message")
    
    if not user_message:
        return {"error": "Vui lòng cung cấp nội dung tin nhắn."}

    start_time = time.perf_counter()
    
    # Thực hiện tác vụ RAG
    response = rag_chain.invoke(user_message)
    
    end_time = time.perf_counter()

    return {
        "answer": response,
        "processing_time_seconds": round(end_time - start_time, 4)
    }