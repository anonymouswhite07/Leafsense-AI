# Root Dockerfile for LeafSense AI Backend
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements
COPY backend/requirements.txt .

# Install dependencies (CPU optimized)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend folder contents to workdir
COPY backend/ .

# Ensure model and solutions are available
# (The root COPY above will have copied everything inside 'backend' to '/app')

EXPOSE 8000

CMD ["python", "main.py"]
