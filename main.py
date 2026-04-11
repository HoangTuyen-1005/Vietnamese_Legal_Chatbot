from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import time

# Import các class mà Thành viên 2 và 3 đã viết
from hybrid_search import HybridSearch
from reranker import Reranker
from generator import LocalLLMGenerator

# 1. Khởi tạo FastAPI Server
app = FastAPI(title="Vietnamese Legal RAG API", version="1.0")

# 2. Tải các AI Model vào RAM/VRAM ngay khi bật Server (Chỉ tải 1 lần)
print("🚀 ĐANG KHỞI ĐỘNG SERVER VÀ TẢI MÔ HÌNH...")
searcher = HybridSearch()
reranker = Reranker()
generator = LocalLLMGenerator() # Đã dùng bản Qwen 1.5B Instruct siêu mượt
print("✅ TẤT CẢ MÔ HÌNH ĐÃ SẴN SÀNG!")

# 3. Định nghĩa cấu trúc dữ liệu đầu vào
class ChatRequest(BaseModel):
    query: str
    session_id: str = "default_user" # Dùng sau này để lưu lịch sử chat

# 4. Định nghĩa API Endpoint chính
@app.post("/api/chat")
async def chat_with_lawyer(request: ChatRequest):
    try:
        start_time = time.time()
        
        # BƯỚC 1: Tìm kiếm Hybrid (Thành viên 2)
        raw_results = searcher.search(request.query, top_k=20)
        
        # BƯỚC 2: Rerank (Thành viên 2)
        top_docs = reranker.rerank(request.query, raw_results, top_n=3)
        
        # BƯỚC 3: Kéo toàn văn ngữ cảnh (Thành viên 2)
        full_contexts = [reranker.get_full_acticle_context(searcher.qdrant, doc) for doc in top_docs]
        
        # BƯỚC 4: Tạo câu trả lời (Thành viên 3)
        final_answer = generator.generate_answer(request.query, full_contexts)
        
        process_time = time.time() - start_time
        
        return {
            "status": "success",
            "query": request.query,
            "answer": final_answer,
            "sources": [{"so_hieu": doc.get("so_hieu"), "dieu": doc.get("dieu")} for doc in top_docs], # Lấy trực tiếp thông tin
            "process_time_seconds": round(process_time, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")

# Chạy Server cục bộ (Chỉ dùng khi test trực tiếp file này)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)