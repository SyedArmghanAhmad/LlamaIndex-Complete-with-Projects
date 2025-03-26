FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements from your local environment
COPY requirements_local.txt .

# Install exact versions from your working local setup
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements_local.txt && \
    python -c "from llama_index.embeddings.huggingface import HuggingFaceEmbedding; print('✓ Imports verified')"

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app_draft.py"]