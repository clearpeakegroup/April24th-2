import gradio as gr
import requests
import os
import json
import numpy as np
import time
import asyncio
import websockets

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
JWT_TOKEN = os.getenv("JWT_TOKEN", "test-token")  # Replace with real token logic
HEADERS = {"Authorization": f"Bearer {JWT_TOKEN}"}

# --- Data Tab ---
def upload_and_preview(file):
    if file is None or not file.name.endswith(".zst"):
        return "Please upload a .zst file.", None
    try:
        files = {"file": (file.name, file.read(), "application/octet-stream")}
        resp = requests.post(f"{BACKEND_URL}/upload", files=files, headers=HEADERS)
        if resp.status_code == 200:
            data = resp.json()
            status = f"Uploaded: {data.get('filename')} ({data.get('rows', 0)} rows)"
            # Optionally, fetch a preview from backend if implemented
            return status, None
        else:
            return f"Upload failed: {resp.text}", None
    except Exception as e:
        return f"Error: {e}", None

data_tab = gr.Tab(
    gr.Interface(
        fn=upload_and_preview,
        inputs=gr.File(label="Upload DBN ZSTD File (.zst)"),
        outputs=[gr.Textbox(label="Status"), gr.Dataframe(label="Sample Ticks")],
        title="Data Upload & Preview"
    ),
    label="Data"
)

# --- Train Tab ---
def launch_training(instruments, algo, timesteps):
    if not instruments:
        return "Select at least one instrument."
    if timesteps < 1000:
        return "Timesteps too low."
    params = {"instruments": instruments, "algo": algo, "timesteps": timesteps}
    try:
        resp = requests.post(f"{BACKEND_URL}/train", json=params, headers=HEADERS)
        if resp.status_code == 200:
            data = resp.json()
            task_id = data.get("task_id")
            status = f"Training started (task: {task_id})..."
            # Poll for result
            for _ in range(30):
                time.sleep(2)
                poll = requests.get(f"{BACKEND_URL}/task_status/{task_id}", headers=HEADERS)
                if poll.status_code == 200:
                    poll_data = poll.json()
                    if poll_data["status"] == "SUCCESS":
                        return f"Training complete: {poll_data['result']}"
                    elif poll_data["status"] == "FAILURE":
                        return f"Training failed: {poll_data['result']}"
            return status + " (still running...)"
        else:
            return f"Train failed: {resp.text}"
    except Exception as e:
        return f"Error: {e}"

train_tab = gr.Tab(
    gr.Interface(
        fn=launch_training,
        inputs=[
            gr.CheckboxGroup(["MES", "NQ", "ES", "RTY", "QQQ"], label="Instruments"),
            gr.Radio(["ppo", "a2c"], label="Algorithm"),
            gr.Number(value=10000, label="Timesteps", precision=0)
        ],
        outputs=gr.Textbox(label="Status"),
        title="Train Agent"
    ),
    label="Train"
)

# --- Backtest Tab ---
def run_backtest(model):
    if not model:
        return "Select a model.", None
    params = {"model": model}
    try:
        resp = requests.post(f"{BACKEND_URL}/backtest", json=params, headers=HEADERS)
        if resp.status_code == 200:
            data = resp.json()
            task_id = data.get("task_id")
            status = f"Backtest started (task: {task_id})..."
            # Poll for result
            for _ in range(30):
                time.sleep(2)
                poll = requests.get(f"{BACKEND_URL}/task_status/{task_id}", headers=HEADERS)
                if poll.status_code == 200:
                    poll_data = poll.json()
                    if poll_data["status"] == "SUCCESS":
                        result = poll_data["result"]
                        # Assume result contains equity curve as a list
                        equity_curve = result.get("equity_curve", np.cumsum(np.random.randn(100)).tolist())
                        return f"Backtest complete for {model}", equity_curve
                    elif poll_data["status"] == "FAILURE":
                        return f"Backtest failed: {poll_data['result']}", None
            return status + " (still running...)", None
        else:
            return f"Backtest failed: {resp.text}", None
    except Exception as e:
        return f"Error: {e}", None

backtest_tab = gr.Tab(
    gr.Interface(
        fn=run_backtest,
        inputs=gr.Dropdown(["ppo_final", "a2c_final"], label="Model"),
        outputs=[gr.Textbox(label="Status"), gr.LinePlot(label="Equity Curve")],
        title="Backtest"
    ),
    label="Backtest"
)

# --- ForwardTest Tab ---
def run_forwardtest():
    params = {}
    try:
        resp = requests.post(f"{BACKEND_URL}/forwardtest", json=params, headers=HEADERS)
        if resp.status_code == 200:
            data = resp.json()
            task_id = data.get("task_id")
            status = f"Forward test started (task: {task_id})..."
            # Poll for result
            for _ in range(30):
                time.sleep(2)
                poll = requests.get(f"{BACKEND_URL}/task_status/{task_id}", headers=HEADERS)
                if poll.status_code == 200:
                    poll_data = poll.json()
                    if poll_data["status"] == "SUCCESS":
                        result = poll_data["result"]
                        equity_curve = result.get("equity_curve", np.cumsum(np.random.randn(100)).tolist())
                        return f"Forward test complete", equity_curve
                    elif poll_data["status"] == "FAILURE":
                        return f"Forward test failed: {poll_data['result']}", None
            return status + " (still running...)", None
        else:
            return f"Forward test failed: {resp.text}", None
    except Exception as e:
        return f"Error: {e}", None

forward_tab = gr.Tab(
    gr.Interface(
        fn=run_forwardtest,
        inputs=[],
        outputs=[gr.Textbox(label="Status"), gr.LinePlot(label="Equity Curve")],
        title="Forward Test"
    ),
    label="ForwardTest"
)

# --- Live Tab ---
def live_status():
    # Placeholder: In production, connect to backend WebSocket for real-time updates
    return np.random.randn(), np.random.randn(10, 2)

def send_manual_order(symbol, qty, side):
    # Placeholder: In production, POST to backend to send manual order
    return f"Order sent: {side} {qty} {symbol}"

live_tab = gr.Tab(
    gr.Blocks(
        lambda: [
            gr.Textbox(label="Real-Time PnL", value="0.0"),
            gr.Dataframe(label="Order Book", value=np.zeros((10, 2))),
            gr.Row([
                gr.Dropdown(["MES", "NQ", "ES", "RTY", "QQQ"], label="Symbol"),
                gr.Number(value=1, label="Qty", precision=0),
                gr.Radio(["buy", "sell"], label="Side"),
                gr.Button("Send Order", elem_id="send_order_btn")
            ]),
            gr.Textbox(label="Order Status", value="")
        ],
        title="Live Trading"
    ),
    label="Live"
)

def ws_live_updates():
    ws_url = BACKEND_URL.replace("http", "ws") + "/ws"
    async def stream():
        pnl_history = []
        try:
            async with websockets.connect(ws_url) as websocket:
                yield "Connected", None, None, []
                for _ in range(100):
                    msg = await websocket.recv()
                    data = json.loads(msg)
                    tick = data.get("tick")
                    pnl = data.get("pnl")
                    pnl_history.append(pnl)
                    yield "Connected", tick, pnl, pnl_history.copy()
                    await asyncio.sleep(1)
        except Exception as e:
            yield f"Error: {e}", None, None, []
    return asyncio.run(stream())

with gr.Blocks() as ws_live_ui:
    gr.Markdown("## Live Trading (Real-Time WebSocket)")
    gr.Markdown("Click 'Start Stream' to receive real-time Tick and PnL updates from the backend via WebSocket.")
    status = gr.Textbox(label="Connection Status", interactive=False)
    with gr.Row():
        tick = gr.Number(label="Tick", precision=4, interactive=False)
        pnl = gr.Number(label="PnL", precision=4, interactive=False)
    pnl_plot = gr.LinePlot(label="PnL History", interactive=False)
    start_btn = gr.Button("Start Stream")

    def start_stream():
        return ws_live_updates()

    start_btn.click(start_stream, inputs=[], outputs=[status, tick, pnl, pnl_plot])

ws_live_tab = gr.Tab(ws_live_ui, label="Live (WS)")

demo = gr.TabbedInterface(
    [data_tab, train_tab, backtest_tab, forward_tab, live_tab, ws_live_tab],
    title="FinRL Platform Control"
)

def main():
    demo.launch()

if __name__ == "__main__":
    main() 