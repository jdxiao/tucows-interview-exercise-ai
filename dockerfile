FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --timeout 120 --retries 5


# Copy project files
COPY src/ src/
COPY data/ data/

# Download TinyLlama model
RUN mkdir -p /app/models
RUN curl -L -o /app/models/TinyLlama-1.1B-32k-f16.gguf \
    https://huggingface.co/andrijdavid/TinyLlama-1.1B-32k-GGUF/resolve/main/TinyLlama-1.1B-32k-f16.gguf

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
