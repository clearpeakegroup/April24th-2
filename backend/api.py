from fastapi import FastAPI, File, UploadFile, WebSocket, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import StreamingResponse
import uvicorn
import os
import jwt
import io
import numpy as np
import asyncio

app = FastAPI(title="FinRL API Gateway")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OAuth2/JWT stub
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")
SECRET_KEY = "dev-secret"

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

# --- REST Endpoints ---
@app.post("/upload")
async def upload_data(file: UploadFile = File(...), user=Depends(get_current_user)):
    # Save file and trigger backend task (stub)
    content = await file.read()
    return {"filename": file.filename, "size": len(content)}

@app.post("/train")
async def train_agent(params: dict, user=Depends(get_current_user)):
    # Trigger Celery task (stub)
    return {"status": "training started", "params": params}

@app.post("/backtest")
async def backtest(params: dict, user=Depends(get_current_user)):
    return {"status": "backtest started", "params": params}

@app.post("/forwardtest")
async def forwardtest(params: dict, user=Depends(get_current_user)):
    return {"status": "forward test started", "params": params}

@app.post("/live")
async def live_exec(params: dict, user=Depends(get_current_user)):
    return {"status": "live execution started", "params": params}

# --- WebSocket for real-time updates ---
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    for i in range(100):
        # Simulate real-time tick and PnL
        await ws.send_json({"tick": np.random.randn(), "pnl": np.random.randn()})
        await asyncio.sleep(1)
    await ws.close()

# --- HTTPS self-signed cert for dev (uvicorn CLI) ---
# To run: uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000 --ssl-keyfile=certs/key.pem --ssl-certfile=certs/cert.pem

if __name__ == "__main__":
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=True) 