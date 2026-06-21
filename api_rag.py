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
from langchain_core.runnables import RunnableLambda

# --- CẤU HÌNH SEMAPHORE (Hàng đợi xử lý) ---
# Chỉ cho phép tối đa 3 request/1 worker(lõi cpu) xử lý đồng thời, các request khác sẽ đợi
MAX_CONCURRENT_REQUESTS = 3
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

vectorstore = None
llm = None
retriever = None
rag_chain = None

async def check_concurrency():
    """Dependency kiểm tra số lượng request, nếu đầy sẽ tự đưa vào hàng đợi."""
    async with request_semaphore:
        yield

# Load các biến môi trường
load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
SALES_PROMPT = """
Bạn là một nhân viên bán hàng chuyên nghiệp.

QUY TẮC CỐT LÕI:

1. Chỉ được trả lời dựa trên thông tin trong tài liệu.
2. Nếu tài liệu không chứa câu trả lời thì trả lời đúng:
'Dạ để em hỏi lại sếp'
3. Không được suy luận.
4. Không được sử dụng kiến thức bên ngoài.
5. Luôn xưng hô dạ, em.

Thông tin tài liệu:

{context}

Câu hỏi khách hàng:

{question}
"""

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

def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore, llm, rag_chain, retriever

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
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.2
    })

    

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
    prompt = ChatPromptTemplate.from_template(SALES_PROMPT)
    rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
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
    # debug
    docs = retriever.invoke(standalone_question)

    print("\n=== RETRIEVED DOCS ===")
    for i, doc in enumerate(docs):
        print(f"\nDOC {i+1}")
        print(doc.page_content[:500])
    # end debug

    t1 = time.perf_counter()
    answer = rag_chain.invoke(standalone_question)
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
