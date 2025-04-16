import gradio as gr
import requests
import os
import json
import numpy as np

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# --- Data Tab ---
def upload_and_preview(file):
    if file is None or not file.name.endswith(".zst"):
        return "Please upload a .zst file.", None
    # Simulate preview: In production, send to backend for parsing
    return f"Uploaded: {file.name}", np.random.randn(10, 7)

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
    # Simulate backend call
    # In production, POST to backend to launch Celery task
    return f"Training started: {algo} on {instruments} for {timesteps} steps."

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
    # Simulate equity curve
    return f"Backtest complete for {model}", np.cumsum(np.random.randn(100))

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
    # Simulate forward test
    return "Forward test running...", np.cumsum(np.random.randn(100))

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
    # Simulate real-time PnL and order book
    return np.random.randn(), np.random.randn(10, 2)

def send_manual_order(symbol, qty, side):
    # Simulate sending manual order
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

demo = gr.TabbedInterface(
    [data_tab, train_tab, backtest_tab, forward_tab, live_tab],
    title="FinRL Platform Control"
)

def main():
    demo.launch()

if __name__ == "__main__":
    main() 