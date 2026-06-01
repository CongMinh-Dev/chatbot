# -*- coding: utf-8 -*-
from fastapi import FastAPI
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from langchain_ollama import OllamaEmbeddings
from contextlib import asynccontextmanager

WORKING_DIR = "./lightrag_db"
rag = None

async def custom_ollama_embed(texts, **kwargs):
    return await ollama_embed(texts, model="embeddinggemma", **kwargs)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="gemma4-local",
        embedding_func=custom_ollama_embed
    )
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/api/chat")
async def chat(request: dict):
    user_message = request.get("message")
    # Truy vấn chế độ local
    response = await rag.aquery(user_message, param=QueryParam(mode="local"))
    return {"answer": response}