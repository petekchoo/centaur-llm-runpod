FROM python:3.10

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
COPY handler.py .

RUN pip install --upgrade pip && pip install -r requirements.txt

# 🛠️ Temporary shell for debugging
CMD [ "bash" ]