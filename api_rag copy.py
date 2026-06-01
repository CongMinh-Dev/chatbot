# -*- coding: utf-8 -*-
import time
import os
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

WORKING_DIR = "./lightrag_db"

# Khai báo các biến toàn cục cho hệ thống
rag = None
cache_store = None

# ==========================================
# 🛠️ QUẢN LÝ VÒNG ĐỜI ỨNG DỤNG (LIFESPAN)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag, cache_store
    
    print("\n🧠 [KHỞI ĐỘNG] Đang thiết lập cấu hình mạng lưới Đồ thị tri thức LightRAG...")
    
    # 1. Khởi tạo thực thể LightRAG nguyên bản theo đúng file cấu trúc lớp lightrag.py
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="gemma4-local:latest",
        embedding_func=ollama_embed
    )
    
    # Gán cấu hình phân đoạn trực tiếp lên thuộc tính thực thể
    rag.chunk_size = 500
    rag.chunk_overlap = 100
    
    # 2. Kích hoạt bộ lưu trữ database ngầm của LightRAG (Bắt buộc với cấu trúc mới)
    print("⚙️ Đang kích hoạt các phân vùng lưu trữ database của LightRAG...")
    await rag.initialize_storages()
    
    # 3. Kết nối hoặc tạo mới kho lưu bộ nhớ đệm Semantic Cache (ChromaDB)
    print("💾 Đang kết nối tới kho lưu bộ nhớ đệm ngữ nghĩa tại ./chroma_cache...")
    cache_embeddings = OllamaEmbeddings(model="embeddinggemma")
    cache_store = Chroma(
        persist_directory="./chroma_cache", 
        embedding_function=cache_embeddings
    )
    
    print("✅ Hệ thống LightRAG & Semantic Cache đã sẵn sàng phục vụ!\n")
    yield
    print("\n🛑 [ĐÓNG] Đang tắt các kết nối hệ thống server...")

# ==========================================
# 🚀 KHỞI TẠO FASTAPI APP
# ==========================================
app = FastAPI(
    title="Sầu Riêng Chatbot LightRAG API + Semantic Cache",
    description="API Server kết hợp Đồ thị tri thức và Semantic Cache tối ưu tốc độ",
    version="2.0.0",
    lifespan=lifespan
)

# Cấu hình CORS để bảo mật mạng và cho phép kết nối Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 🛠️ CÁC HÀM TIỆN ÍCH XỬ LÝ SEMANTIC CACHE
# ==========================================
def get_semantic_cache(user_query: str):
    """Tìm kiếm câu hỏi tương tự trong bộ nhớ đệm để phản hồi ngay lập tức nếu khớp ngữ nghĩa"""
    try:
        if cache_store is None:
            return None, None
        results = cache_store.similarity_search_with_score(user_query, k=1)
        if results:
            doc, score = results[0]
            if score < 0.4:  # Ngưỡng khoảng cách ngữ nghĩa (Càng nhỏ càng chính xác)
                return doc.metadata.get("answer"), score
    except Exception as e:
        print(f"⚠️ Không thể đọc hệ thống Semantic Cache: {e}")
    return None, None

def save_semantic_cache(user_query: str, ai_answer: str):
    """Lưu cặp câu hỏi - câu trả lời mới vào bộ nhớ đệm ChromaDB"""
    try:
        if cache_store is not None:
            cache_store.add_texts(
                texts=[user_query],
                metadatas=[{"answer": ai_answer}]
            )
            print("💾 [CACHE] Đã lưu cặp phản hồi thành công vào ./chroma_cache")
    except Exception as e:
        print(f"⚠️ Không thể ghi dữ liệu mới vào Semantic Cache: {e}")

# ==========================================
# 📩 ĐỊNH NGHĨA ĐƯỜNG DẪN ENDPOINT API CHÍNH
# ==========================================
class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    user_message = request.message
    if not user_message or not user_message.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống")
        
    start_total = time.time()
    try:
        print("\n" + "="*50)
        print(f"📩 Nhận câu hỏi từ người dùng: {user_message}")
        
        # BƯỚC 1: Kiểm tra bộ nhớ đệm Cache trước tiên
        cached_answer, similarity_score = get_semantic_cache(user_message)
        if cached_answer:
            total_duration = time.time() - start_total
            print(f"⚡ [CACHE HIT] Tìm thấy câu hỏi tương tự! Trả kết quả siêu tốc (Score: {similarity_score:.4f})")
            return {
                "status": "success",
                "query": user_message,
                "answer": cached_answer,
                "debug": {
                    "cache_hit": True, 
                    "similarity_score": round(float(similarity_score), 4),
                    "total_time_seconds": round(total_duration, 2)
                }
            }
            
        # BƯỚC 2: Nếu chưa có trong cache, tiến hành quét Đồ thị Tri thức LightRAG
        print("🐢 [CACHE MISS] Đang tra cứu chuyên sâu trên Đồ thị Tri thức LightRAG...")
        start_lightrag = time.time()
        
        # Gọi hàm aquery xử lý không đồng bộ bất đồng bộ chuẩn của thư viện
        response_text = await rag.aquery(
            user_message,
            param=QueryParam(mode="local") # Chế độ local tối ưu cho việc truy xuất chi tiết quy trình cụ thể
        )
        
        lightrag_duration = time.time() - start_lightrag
        
        # BƯỚC 3: Ghi kết quả mới bóc tách được vào kho Cache cho lần sau
        save_semantic_cache(user_message, response_text)
        
        # BƯỚC 4: Tổng kết thời gian chạy của hệ thống
        total_duration = time.time() - start_total
        print(f"✨ Xử lý thành công câu hỏi mới trong: {total_duration:.4f} giây")
        print("="*50 + "\n")
        
        return {
            "status": "success",
            "query": user_message,
            "answer": response_text,
            "debug": {
                "cache_hit": False,
                "lightrag_time_seconds": round(lightrag_duration, 2),
                "total_time_seconds": round(total_duration, 2)
            }
        }
    except Exception as e:
        print(f"❌ [LỖI HỆ THỐNG API]: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Endpoint kiểm tra trạng thái hoạt động của Server"""
    return {
        "status": "healthy", 
        "engine": "LightRAG Engine (Local Mode)", 
        "models_in_use": {
            "llm": "qwen2.5:3b", 
            "embedding": "embeddinggemma"
        }
    }