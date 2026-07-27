FROM python:3.10-slim

# Cài đặt các công cụ hệ thống cần thiết cho PostgreSQL client & tini
RUN apt-get update && apt-get install -y \
    tini \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["/usr/bin/tini", "--"]

# Chạy Uvicorn với 3 worker chịu tải cao
CMD ["uvicorn", "api_rag:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]