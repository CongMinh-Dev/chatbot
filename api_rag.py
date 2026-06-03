# -*- coding: utf-8 -*-
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
        # tôi đã sửa model embedding mặc định trong thư viện thành cái của tôi rồi, vì sửa ở ngoài code kiểu gì cũng không làm được
        embedding_func=ollama_embed, 
        addon_params={
            "language": "Vietnamese",
            "entity_relationship_graph_type": "default"
        }
    )
    yield

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
        # Truy vấn chế độ local (hoặc mode khác truyền lên từ client)
        response = await rag.aquery(user_message, param=QueryParam(mode=user_mode))
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")