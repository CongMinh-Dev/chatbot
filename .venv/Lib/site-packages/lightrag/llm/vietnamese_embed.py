"""
Vietnamese Embedding via HuggingFace Inference API
Model: BAAI/bge-m3
"""

import os
import numpy as np
from typing import Optional, List
from huggingface_hub import InferenceClient
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from lightrag.utils import wrap_embedding_func_with_attrs, logger

DEFAULT_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
MAX_RETRIES = 3

class VietnameseEmbeddingError(Exception):
    pass

class VietnameseEmbeddingRateLimitError(VietnameseEmbeddingError):
    pass

class VietnameseEmbeddingConnectionError(VietnameseEmbeddingError):
    pass

class VietnameseEmbeddingTimeoutError(VietnameseEmbeddingError):
    pass

def get_api_token() -> str:
    token = os.environ.get("EMBEDDING_BINDING_API_KEY") or os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HuggingFace API token required")
    return token

@wrap_embedding_func_with_attrs(embedding_dim=EMBEDDING_DIM)
@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_exception_type((VietnameseEmbeddingRateLimitError, VietnameseEmbeddingConnectionError, VietnameseEmbeddingTimeoutError)))
async def vietnamese_embed(texts: List[str], model_name: str = DEFAULT_MODEL, token: Optional[str] = None) -> np.ndarray:
    if token is None:
        token = get_api_token()
    
    # Override AITeamVN/Vietnamese_Embedding to use BAAI/bge-m3 (base model that supports Inference API)
    # AITeamVN model doesn't support feature-extraction API, only sentence-similarity
    if "AITeamVN" in model_name or "Vietnamese_Embedding" in model_name:
        logger.warning(f"AITeamVN/Vietnamese_Embedding doesn't support Inference API feature-extraction. Using BAAI/bge-m3 instead.")
        model_name = DEFAULT_MODEL
    
    logger.info(f"Calling HuggingFace Inference API for {len(texts)} texts using {model_name}")
    
    try:
        client = InferenceClient(api_key=token)
        embeddings_list = []
        
        for text in texts:
            try:
                embedding = client.feature_extraction(text, model=model_name)
                emb_array = np.array(embedding, dtype=np.float32) if not isinstance(embedding, np.ndarray) else embedding
                if emb_array.ndim > 1:
                    emb_array = emb_array.flatten()
                embeddings_list.append(emb_array)
            except Exception as e:
                error_msg = str(e).lower()
                if 'rate limit' in error_msg or '429' in error_msg:
                    raise VietnameseEmbeddingRateLimitError(f"Rate limit: {e}") from e
                elif 'timeout' in error_msg:
                    raise VietnameseEmbeddingTimeoutError(f"Timeout: {e}") from e
                else:
                    raise VietnameseEmbeddingConnectionError(f"API error: {e}") from e
        
        embeddings_np = np.stack(embeddings_list, axis=0)
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        embeddings_np = embeddings_np / np.maximum(norms, 1e-9)
        
        logger.info(f"✅ Generated embeddings: shape {embeddings_np.shape}")
        return embeddings_np
        
    except (VietnameseEmbeddingRateLimitError, VietnameseEmbeddingConnectionError, VietnameseEmbeddingTimeoutError):
        raise
    except Exception as e:
        raise VietnameseEmbeddingConnectionError(f"Embedding failed: {e}") from e

@wrap_embedding_func_with_attrs(embedding_dim=EMBEDDING_DIM)
async def vietnamese_embedding_func(texts: List[str]) -> np.ndarray:
    return await vietnamese_embed(texts)

async def vietnamese_embed_texts(texts: List[str]) -> np.ndarray:
    return await vietnamese_embedding_func(texts)
