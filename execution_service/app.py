from fastapi import FastAPI
from uuid import uuid4
from shared.models import Order

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/broker/order")
def create_order(order: Order):
    return {"id": str(uuid4())} 