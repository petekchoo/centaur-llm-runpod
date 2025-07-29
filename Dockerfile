FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy local files to the container
COPY requirements.txt .
COPY handler.py .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# RunPod expects this command
CMD ["python3", "-m", "runpod"]