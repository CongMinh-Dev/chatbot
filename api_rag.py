import time
import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
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
retriever = None
rag_chain = None
rewrite_chain = None
embeddings = None

async def check_concurrency():
    """Dependency kiểm tra số lượng request, nếu đầy sẽ tự đưa vào hàng đợi."""
    async with request_semaphore:
        yield

# Load các biến môi trường
load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SALES_PROMPT = """
Bạn là một nhân viên bán hàng chuyên nghiệp.

QUY TẮC CỐT LÕI:

1. Chỉ được trả lời dựa trên thông tin trong tài liệu.
2. Nếu tài liệu có câu trả lời thì cứ trả lời dạ kèm câu trả lời.
3. Nếu tài liệu không chứa câu trả lời thì trả lời chính xác là:'Dạ để em hỏi lại sếp'.
4. Không được suy luận.
5. Không được sử dụng kiến thức bên ngoài.
6. Luôn xưng hô dạ, em.

Thông tin tài liệu:

{context}

Câu hỏi khách hàng:

{question}
"""

rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Bạn là bộ chuyển đổi câu hỏi.

        Nhiệm vụ:
        - Dựa trên lịch sử hội thoại để viết lại câu hỏi cuối thành một câu hỏi độc lập, đầy đủ ngữ cảnh.
        - Không trả lời câu hỏi.
        - Không giải thích.
        - Chỉ sửa lỗi chính tả, sửa ngữ pháp, sửa nghĩa của câu. sửa sao cho phù hợp.
        - Thay thế các đại từ tham chiếu (ví dụ: "nó", "đó", "cái này", "họ",...) bằng thực thể tương ứng nếu có thể suy ra từ lịch sử.

        Đầu ra:
        - Chỉ trả về đúng một câu hỏi đã viết lại.
        - Không thêm bất kỳ nội dung nào khác.
        """
    ),
    (
        "human",
        """
        Lịch sử:
        {history}

        Câu hỏi:
        {question}
        """
    )
])



def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore, rag_chain, retriever, rewrite_chain, embeddings

    # 1. Sử dụng OllamaEmbeddings (Đảm bảo base_url đúng IP LXC của bạn)
    # embeddings = NVIDIAEmbeddings(
    #     model="nvidia/nv-embed-v1", 
    #     api_key=NVIDIA_API_KEY
    # )
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
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
    # llm = ChatOpenAI(
    #     model="deepseek-ai/deepseek-v4-flash",
    #     base_url="https://integrate.api.nvidia.com/v1",
    #     api_key=NVIDIA_API_KEY,
    #     temperature=0,
    #     model_kwargs={
    #         "max_tokens": 16384,
    #         "top_p": 0.95,
    #         "extra_body": {
    #             "chat_template_kwargs": {
    #                 "thinking": False
    #             }
    #         }
    #     }
    # )
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
        max_output_tokens=512,
    )

    llmVietLaiCauHoi = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=GOOGLE_API_KEY,
        temperature=1,
        max_output_tokens=512,
    )

    # llmVietLaiCauHoi = ChatOpenAI(
    #     model="qwen/qwen3.5-122b-a10b", # Đổi model
    #     base_url="https://integrate.api.nvidia.com/v1",
    #     api_key=NVIDIA_API_KEY,
    #     temperature=0.60, # Theo cấu hình yêu cầu
    #     model_kwargs={
    #         "extra_body": {
    #             "max_tokens": 16384, # Theo cấu hình yêu cầu
    #             "top_p": 0.95,     # Theo cấu hình yêu cầu
    #         }
    #     }
    # )

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
    print(f'rag chain: {rag_chain}')
    rewrite_chain = (
        rewrite_prompt
        | llmVietLaiCauHoi
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

    # viết lại câu hỏi
    start_time = time.perf_counter()
    standalone_question = rewrite_chain.invoke({
        "history": history_text,
        "question": latest_question
    })
    t0_1 = time.perf_counter()
    
    # debug
    print("\n===================================================================")
    print("LATEST QUESTION:")
    print(latest_question)
    print("\nSTANDALONE QUESTION:")
    print(standalone_question)
    print("=====================================================================\n")

    # RETRIEVER
    t0_2 = time.perf_counter()
    print(type(standalone_question))
    print(isinstance(standalone_question, str))
    print(type(str(standalone_question)))
    print(repr(standalone_question))
    vec = embeddings.embed_query("Xin chào")
    print(len(vec))
    # docs = retriever.invoke(standalone_question)
    docs = retriever.invoke("Có những sản phẩm nào thế?")

    print("\n=== RETRIEVED DOCS ===")
    for i, doc in enumerate(docs):
        print(f"\nDOC {i+1}")
        print(doc.page_content[:500])
    # end debug

    t1 = time.perf_counter()
    print("Before invoke===")
    print("Global embeddings:", id(embeddings))
    print("Retriever embeddings:", id(retriever.vectorstore._embedding_function))
    print("Same object:", embeddings is retriever.vectorstore._embedding_function)
    answer = rag_chain.invoke(standalone_question)
    # answer = llm.invoke("Xin chào")
    print("After invoke")
    t2 = time.perf_counter()

    print(f"Model response: {answer}")
    print(f"viết lại câu hỏi seconds: {round(t0_1 - start_time, 4)}")
    print(f"generation seconds: {round(t2 - t1, 4)}")

    return {
        "answer": answer,
        "timing": {
            "retrieval_seconds": round(t1 - t0_2, 4),
            "generation_seconds": round(t2 - t1, 4),
            "total_seconds": round(t2 - start_time, 4)
        }
    }
