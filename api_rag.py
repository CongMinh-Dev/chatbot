# -*- coding: utf-8 -*-
import time  # <--- Thêm thư viện time để đo thời gian
from fastapi import FastAPI, HTTPException, Body
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from contextlib import asynccontextmanager

WORKING_DIR = "./lightrag_db"
rag = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="gemma4-local:latest",
        # llm_model_name="qwen2.5:3b",
        embedding_func=ollama_embed, 
        addon_params={
            "language": "Vietnamese",
            "entity_relationship_graph_type": "default"
        }
    )
    
    # Khởi tạo các kho lưu trữ dữ liệu khi app start
    print("Đang khởi tạo các kho lưu trữ dữ liệu (Storages)...")
    await rag.initialize_storages()
    print("Khởi tạo Storages thành công!")
    
    yield
    
    # Đóng lưu trữ an toàn khi tắt app
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
            "message": "Hãy tóm tắt nội dung tài liệu giúp tôi",
            "mode": "local"
        }
    )
):
    if not rag:
        raise HTTPException(status_code=500, detail="LightRAG chưa được khởi tạo.")
        
    user_message = request.get("message")
    user_mode = request.get("mode", "local")
    
    if not user_message:
        raise HTTPException(status_code=400, detail="Thiếu trường 'message' trong request body.")
        
    try:
        # 1. Bắt đầu bấm giờ
        start_time = time.time()
        
        # Gọi RAG lấy câu trả lời
        response = await rag.aquery(user_message, param=QueryParam(mode=user_mode))
        
        # 2. Tính toán thời gian đã trôi qua (đơn vị: giây)
        execution_time = time.time() - start_time
        
        # 3. Trả về câu trả lời kèm theo thời gian xử lý (làm tròn 2 chữ số thập phân)
        return {
            "answer": response,
            "execution_time_seconds": round(execution_time, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")
    
print("vào kiểm tra:-------------------------------http://localhost:8000/docs")