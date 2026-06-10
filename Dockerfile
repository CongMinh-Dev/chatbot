# 1. Chọn hệ điều hành cơ bản
FROM python:3.10

# 2. CÀI CÔNG CỤ HỆ THỐNG (Dùng apt-get - Không dùng requirements.txt ở đây)
RUN apt-get update && apt-get install -y tini && rm -rf /var/lib/apt/lists/*

# 3. CÀI THƯ VIỆN PYTHON (Dùng pip - Đây là nơi dùng requirements.txt)
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. COPY CODE VÀ CHẠY
COPY . .
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "api_rag:app", "--host", "0.0.0.0", "--port", "8000"]