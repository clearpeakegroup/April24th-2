FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gradio==4.44.1

COPY frontend ./frontend
CMD ["python3", "-u", "frontend/gradio_app.py"]
