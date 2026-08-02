FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . smartlearn-backend/

CMD ["sh", "-c", "uvicorn smartlearn-backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
