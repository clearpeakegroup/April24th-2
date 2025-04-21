# ── build & runtime image
FROM python:3.10-slim

# 1. system libs   (rarely change  → cached)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

# 2. requirements  (huge, cached until you edit requirements.txt)
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. source code   (tiny, re‑copied on each code edit)
COPY . .

CMD ["uvicorn", "entrypoint:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
