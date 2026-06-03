# -*- coding: utf-8 -*-
import time
from fastapi import FastAPI, HTTPException, Body
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from contextlib import asynccontextmanager

WORKING_DIR = "./lightrag_db"
rag = None

# Định nghĩa Prompt đóng vai cho LightRAG
SALES_PROMPT = (
    "Bạn là một nhân viên bán hàng chuyên nghiệp, luôn lịch sự, niềm nở và xưng hô 'dạ', 'em' với khách hàng.\n"
    "QUY TẮC CỐT LÕI:\n"
    "1) Chỉ sử dụng thông tin được cung cấp trong tài liệu (Context) để trả lời khách hàng.\n"
    "2) Nếu câu hỏi của khách hàng nằm ngoài phạm vi tài liệu hoặc tài liệu không có thông tin rõ ràng, "
    "bạn BẮT BUỘC phải trả lời nguyên văn câu này: 'Dạ để em hỏi lại sếp'. Không được tự ý bịa đặt hoặc dùng kiến thức bên ngoài."
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="gemma4-local:latest",
        embedding_func=ollama_embed, 
        addon_params={
            "language": "Vietnamese",
            "entity_relationship_graph_type": "default"
        }
    )
    
    print("Đang khởi tạo các kho lưu trữ dữ liệu (Storages)...")
    await rag.initialize_storages()
    print("Khởi tạo Storages thành công!")
    
    yield
    
    if rag:
        print("Đang đóng các kho lưu trữ dữ liệu...")
        await rag.finalize_storages()
        print("Đã đóng Storages an toàn!")

app = FastAPI(lifespan=lifespan)

@app.post("/api/chat")
async def chat(
    request: dict = Body(
        ..., 
        example={
            "message": "Giá bán sầu riêng loại 1 là bao nhiêu em?",
            "mode": "naive"  # Hãy dùng thử 'naive' để test tốc độ trước nha bạn
        }
    )
):
    if not rag:
        raise HTTPException(status_code=500, detail="LightRAG chưa được khởi tạo.")
        
    user_message = request.get("message")
    user_mode = request.get("mode", "naive")  # Mặc định để naive cho nhanh
    
    if not user_message:
        raise HTTPException(status_code=400, detail="Thiếu trường 'message' trong request body.")
        
    try:
        # Bắt đầu bấm giờ
        start_time = time.time()
        
        # Truy vấn với hệ thống Prompt ép luật của bạn
        response = await rag.aquery(
            user_message,                     # Tham số 1: query
            QueryParam(mode=user_mode),       # Tham số 2: param
            system_prompt=SALES_PROMPT        # Tham số 3: system_prompt
        )
        
        # Tính thời gian chạy
        execution_time = time.time() - start_time
        
        return {
            "answer": response,
            "execution_time_seconds": round(execution_time, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")
    
print("vào kiểm tra:-------------------------------http://localhost:8000/docs")