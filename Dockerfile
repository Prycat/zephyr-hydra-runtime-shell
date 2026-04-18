# Prycat — CLI agent
# Usage: docker compose up
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ollama runs as a separate service — override via OLLAMA_HOST env var
ENV OLLAMA_HOST=http://ollama:11434

CMD ["python", "agent.py"]
