FROM python:3.10-slim

# System dependencies for OpenCV (used by Ultralytics YOLO) and PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Ensure embedding model is present; download fallback if missing
RUN python -c "from pathlib import Path; from sentence_transformers import SentenceTransformer; p = Path('/app/models/static-similarity-mrl-multilingual-v1'); (p.exists() and any(p.iterdir())) or SentenceTransformer('sentence-transformers/static-similarity-mrl-multilingual-v1').save(str(p))"

ENV PYTHONUNBUFFERED=1

CMD ["python", "process_pdfs_1b.py"]
