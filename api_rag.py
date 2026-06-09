import time
from fastapi import FastAPI, Body
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from contextlib import asynccontextmanager

# Định nghĩa Prompt bán hàng
SALES_PROMPT = (
    "Bạn là một nhân viên bán hàng chuyên nghiệp, luôn lịch sự, niềm nở và xưng hô 'dạ', 'em' với khách hàng.\n"
    "QUY TẮC CỐT LÕI:\n"
    "1) Chỉ trả lời dựa trên thông tin có trong tài liệu.\n"
    "2) Khi khách hàng hỏi về danh sách sản phẩm theo tiêu chí (giá, công dụng, xuất xứ...), "
    "hãy liệt kê ĐẦY ĐỦ tất cả các sản phẩm tìm thấy trong tài liệu thỏa mãn điều kiện đó. "
    "Tuyệt đối không được bỏ sót hoặc chỉ nêu một sản phẩm.\n"
    "3) Nếu không tìm thấy bất kỳ sản phẩm nào thỏa mãn điều kiện của khách hàng, "
    "bạn bắt buộc phải trả lời đúng nguyên văn câu: 'Dạ để em hỏi lại sếp'.\n"
    "4) Không được tự suy luận, không giải thích quá trình tìm kiếm, chỉ tập trung trả lời trực tiếp vào yêu cầu của khách.\n"
    "5) Khi tìm thấy NHIỀU sản phẩm cùng thỏa mãn tiêu chí khách hỏi (ví dụ: cùng giúp mượt lông), "
    "hãy liệt kê tất cả tên các sản phẩm đó và mô tả ngắn gọn điểm khác biệt (ví dụ: loại thức ăn hạt, loại gel, hay loại pate) "
    "để khách hàng dễ dàng lựa chọn. Tuyệt đối không được trả lời chung chung hoặc chỉ chọn 1 sản phẩm."
)

vectorstore = None
rag_chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore, rag_chain
    # Load embedding model
    embeddings = OllamaEmbeddings(model="bge-m3:latest")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    
    # Sử dụng Gemma4
    llm = ChatOllama(model="gemma4-local:latest", temperature=0)
    
    # Kết hợp Prompt vào template
    template = (SALES_PROMPT + "\n\nContext: {context}\n\nQuestion: {question}")
    prompt = ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/api/chat")
async def chat(
    request: dict = Body(..., example={"message": "Sản phẩm nào giúp mượt lông?"})
):
    user_message = request.get("message")
    
    # Đo thời gian bắt đầu
    start_time = time.perf_counter()
    
    # Thực hiện RAG chain
    response = rag_chain.invoke(user_message)
    
    # Đo thời gian kết thúc
    end_time = time.perf_counter()
    
    return {
        "answer": response,
        "processing_time_seconds": round(end_time - start_time, 4)
    }