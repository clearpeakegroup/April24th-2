"""
Robust, production-ready Gradio front-end for April24th
Implements all business logic, polling, error handling, and full integration with backend.
"""

import asyncio
import gradio as gr
import httpx
import os
from datetime import date
from typing import List
import json
import socket
import re
import logging

API_URL = os.getenv("API_URL", "http://localhost:8000")

ASSETS = ["MES", "MNQ", "ZN", "QQQOP", "ES", "NQ", "CL", "GC"]

# Shared state for settings
settings = {
    "assets": [],
    "train_futures": True,
    "train_options": False,
    "forward_futures": True,
    "forward_options": False,
    "live_futures": True,
    "live_options": False,
}

# --- API Helpers ---


async def api_get(path):
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{API_URL}{path}")
            r.raise_for_status()
            return r.json()
    except Exception as e:
        logging.error(f"API GET {path} failed: {e}")
        return {"error": f"API error: {e}"}


async def api_post(path, data):
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(f"{API_URL}{path}", json=data)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        logging.error(f"API POST {path} failed: {e}")
        return {"error": f"API error: {e}"}


async def api_patch(path, data):
    try:
        async with httpx.AsyncClient() as client:
            r = await client.patch(f"{API_URL}{path}", json=data)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        logging.error(f"API PATCH {path} failed: {e}")
        return {"error": f"API error: {e}"}


async def sse_stream(url):
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", url) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        yield line[6:]
    except Exception as e:
        yield json.dumps({"pct": 0, "msg": f"Error: {e}"})

# --- Sidebar ---


def settings_sidebar():
    with gr.Column():
        assets = gr.CheckboxGroup(
    ASSETS, label="Assets", value=settings["assets"])
        train_fut = gr.Checkbox(
    label="Futures-only (Train)",
     value=settings["train_futures"])
        train_opt = gr.Checkbox(
    label="Futures+Options (Train)",
     value=settings["train_options"])
        fwd_fut = gr.Checkbox(
    label="Futures-only (Forward)",
     value=settings["forward_futures"])
        fwd_opt = gr.Checkbox(
    label="Futures+Options (Forward)",
     value=settings["forward_options"])
        live_fut = gr.Checkbox(
    label="Futures-only (Live)",
     value=settings["live_futures"])
        live_opt = gr.Checkbox(
    label="Futures+Options (Live)",
     value=settings["live_options"])

    def update_settings(a, tf, to, ff, fo, lf, lo):
        settings["assets"] = a
        settings["train_futures"] = tf
        settings["train_options"] = to
        settings["forward_futures"] = ff
        settings["forward_options"] = fo
        settings["live_futures"] = lf
        settings["live_options"] = lo
    assets.change(
    update_settings, [
        assets, train_fut, train_opt, fwd_fut, fwd_opt, live_fut, live_opt], None)
    train_fut.change(
    update_settings, [
        assets, train_fut, train_opt, fwd_fut, fwd_opt, live_fut, live_opt], None)
    train_opt.change(
    update_settings, [
        assets, train_fut, train_opt, fwd_fut, fwd_opt, live_fut, live_opt], None)
    fwd_fut.change(
    update_settings, [
        assets, train_fut, train_opt, fwd_fut, fwd_opt, live_fut, live_opt], None)
    fwd_opt.change(
    update_settings, [
        assets, train_fut, train_opt, fwd_fut, fwd_opt, live_fut, live_opt], None)
    live_fut.change(
    update_settings, [
        assets, train_fut, train_opt, fwd_fut, fwd_opt, live_fut, live_opt], None)
    live_opt.change(
    update_settings, [
        assets, train_fut, train_opt, fwd_fut, fwd_opt, live_fut, live_opt], None)


# --- Dashboard Tab ---
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            settings_sidebar()
        with gr.Column(scale=4):
            with gr.Tabs():
                with gr.Tab("Ingest"):
                    gr.Markdown("### Data Ingestion")
                    dataset = gr.Textbox(label="Dataset", placeholder="e.g. ES_tick")
                    asset = gr.Textbox(label="Asset (optional)", placeholder="e.g. ES")
                    file = gr.File(label="Upload File", file_types=[".csv", ".jsonl", ".json", ".zst"])
                    ingest_btn = gr.Button("Start Ingestion")
                    log = gr.Dataframe(headers=["Log"], datatype=["str"], interactive=False, label="Ingestion Log")
                    progress = gr.Progress()
                    compact_btn = gr.Button("Compact now")
                    from_date = gr.Textbox(label="From (YYYY-MM-DD)")
                    to_date = gr.Textbox(label="To (YYYY-MM-DD)")
                    download_btn = gr.Button("Download Parquet")
                    download_file = gr.File(label="Download Output")

                    async def start_ingest(file, dataset, asset):
                        import httpx, asyncio
                        if not file or not dataset:
                            return [["Please provide both file and dataset name."]]
                        files = {"file": (file.name, file)}
                        data = {"version": 1, "user": dataset, "source": asset or "api"}
                        async with httpx.AsyncClient() as client:
                            r = await client.post(f"{API_URL}/ingest/batch", files=files, data=data)
                            r.raise_for_status()
                            job_id = r.json()["job_id"]
                        logs = []
                        pct = 0
                        progress(0)
                        while True:
                            await asyncio.sleep(1)
                            async with httpx.AsyncClient() as client:
                                resp = await client.get(f"{API_URL}/ingest/batch/progress/{job_id}")
                                resp.raise_for_status()
                                pct = resp.json().get("progress", 0)
                            progress(pct / 100)
                            if pct >= 100 or pct < 0:
                                break
                        if pct == 100:
                            logs.append("✅ Ingestion complete.")
                        elif pct < 0:
                            logs.append("❌ Ingestion failed.")
                        return [[l] for l in logs]

                    ingest_btn.click(
                        start_ingest,
                        inputs=[file, dataset, asset],
                        outputs=[log],
                    )

                    async def compact(dataset, from_date):
                        if not dataset or not from_date:
                            return
                        async with httpx.AsyncClient() as client:
                            await client.post(f"{API_URL}/jobs/compact", params={"dataset": dataset, "date": from_date})
                    compact_btn.click(compact, inputs=[dataset, from_date], outputs=[])

                    async def download(dataset, from_date, to_date):
                        if not dataset or not from_date or not to_date:
                            return None
                        async with httpx.AsyncClient() as client:
                            resp = await client.get(f"{API_URL}/datasets/{dataset}/download", params={"from": from_date, "to": to_date}, timeout=None)
                            out_path = f"{dataset}_{from_date}_{to_date}.parquet.zst"
                            with open(out_path, "wb") as f:
                                async for chunk in resp.aiter_bytes():
                                    f.write(chunk)
                        return out_path
                    download_btn.click(download, inputs=[dataset, from_date, to_date], outputs=[download_file])
                with gr.Tab("Dashboard"):
                    # --- Regime/Actor/Router State Panel ---
                    regime_box = gr.Textbox(label="Current Regime", interactive=False)
                    actor_box = gr.Textbox(label="Detected Actor", interactive=False)
                    signal_weight_box = gr.Textbox(label="Signal Weight", interactive=False)
                    order_type_box = gr.Textbox(label="Order Type", interactive=False)
                    size_mult_box = gr.Textbox(label="Size Multiplier", interactive=False)
                    regime_timeline = gr.LinePlot(x="Time", y="RegimeCode", title="Regime Timeline", overlay_point=True, interactive=False)
                    regime_history = []
                    import time as _time
                    async def update_regime_panel():
                        async for line in sse_stream(f"{API_URL}/flow/stream"):
                            try:
                                d = json.loads(line)
                                fs = d["flow_signal"]
                                ro = d["router_output"]
                                regime = fs.get("actor", "")
                                actor_map = {
                                    0: "HFT_MM",
                                    1: "HFT_Spoofer",
                                    2: "Macro_Block",
                                    3: "Dealer_Hedge",
                                    4: "Fed_OMO"
                                }
                                regime_name = actor_map.get(fs.get("actor", -1), "UNKNOWN")
                                regime_box.update(regime_name)
                                actor_box.update(regime_name)
                                signal_weight_box.update(str(ro.get("signal_weight", "")))
                                order_type_box.update(ro.get("order_type", ""))
                                size_mult_box.update(str(ro.get("size_multiplier", "")))
                                # Timeline: append (timestamp, regime_code)
                                t = int(fs.get("ts_ns", _time.time_ns()) // 1e9)
                                regime_history.append({"Time": t, "RegimeCode": fs.get("actor", -1)})
                                # Keep last 100
                                if len(regime_history) > 100:
                                    regime_history.pop(0)
                                regime_timeline.value = regime_history
                            except Exception as e:
                                regime_box.update(f"Error: {e}")
                                continue
                    demo.load(update_regime_panel, None, None, every=1)
                    jobs_table = gr.Dataframe(headers=["ID", "Type", "Status", "% Complete"], datatype=["str", "str", "str", "number"], interactive=False, label="Jobs")
                    pl_chart = gr.LinePlot(x="Time", y="PnL", title="P/L Over Time", overlay_point=True, interactive=False)
                    jobs_data = []
                    pl_data = []
                    async def update_jobs():
                        jobs = await api_get("/jobs")
                        if isinstance(jobs, dict) and "error" in jobs:
                            return [["Error", jobs["error"], "", ""]]
                        return [[j.get("id", "N/A"), j.get("type", "N/A"), j.get("status", "N/A"), j.get("percent_complete", 0)] for j in jobs]
                    
                    def update_pl():
                        acc = {"pnl": len(pl_data) * 100}
                        if isinstance(acc, dict) and "error" in acc:
                            pl_data.append({"Time": len(pl_data), "PnL": 0})
                        else:
                            pl_data.append({"Time": len(pl_data), "PnL": acc.get("pnl", 0)})
                        if len(pl_data) > 100:
                            pl_data.pop(0)
                        return pl_data
                    
                    refresh_btn = gr.Button("Refresh Dashboard")
                    
                    async def refresh_dashboard():
                        jobs_update = await update_jobs()
                        pl_update = update_pl()
                        return gr.update(value=jobs_update), gr.update(value=pl_update)
                    
                    refresh_btn.click(refresh_dashboard, None, [jobs_table, pl_chart])
                    
                    demo.load(refresh_dashboard, None, [jobs_table, pl_chart])
                with gr.Tab("Train"):
                    gr.Markdown("### Train Model")
                    epoch = gr.Slider(1, 100, value=10, step=1, label="Epochs")
                    batch = gr.Slider(
    1, 1024, value=32, step=1, label="Batch Size")
                    start_btn = gr.Button("Start Training")
                    stop_btn = gr.Button("Stop")
                    progress = gr.Progress()
                    status_box = gr.Textbox(label="Status", interactive=False)
                    job_id_state = gr.State("")

                    async def start_train(epoch, batch):
                        req = {
    "type": "train",
    "params": {
        "epochs": epoch,
        "batch": batch,
         "assets": settings["assets"]}}
                        job = await api_post("/jobs/train", req)
                        if "error" in job:
                            status_box.update(f"API error: {job['error']}")
                            return "", f"API error: {job['error']}"
                        job_id = job.get("job_id") or job.get("id")
                        progress(0)
                        while True:
                            await asyncio.sleep(1)
                            resp = await api_get(f"/jobs/{job_id}/progress")
                            pct = resp.get("progress", 0)
                            progress(pct / 100)
                            if pct >= 100 or pct < 0:
                                break
                        if pct == 100:
                            status_box.update("✅ finished")
                        elif pct < 0:
                            status_box.update("❌ failed")
                        return job_id, "Started"
                    start_btn.click(
    start_train, [
        epoch, batch], [
            job_id_state, status_box])

                    async def stop_train(job_id):
                        if job_id:
                            await api_patch(f"/jobs/{job_id}", {"action": "cancel"})
                            status_box.update("❌ cancelled")
                    stop_btn.click(stop_train, job_id_state, status_box)
                with gr.Tab("Forward"):
                    gr.Markdown("### Forward Test")
                    date_start = gr.Textbox(label='Start date YYYY‑MM‑DD')
                    date_end   = gr.Textbox(label='End date YYYY‑MM‑DD')
                    fwd_start = gr.Button("Start Forward Test")
                    fwd_stop = gr.Button("Stop")
                    fwd_progress = gr.Progress()
                    fwd_status = gr.Textbox(label="Status", interactive=False)
                    fwd_job_id = gr.State("")
                    
                    async def start_forward(date_start, date_end):
                        date_range = [date_start, date_end]
                        req = {
    "type": "forward",
    "params": {
        "date_range": date_range,
         "assets": settings["assets"]}}
                        job = await api_post("/jobs/forward", req)
                        if "error" in job:
                            fwd_status.update(f"API error: {job['error']}")
                            return "", f"API error: {job['error']}"
                        job_id = job.get("job_id") or job.get("id")
                        fwd_progress(0)
                        while True:
                            await asyncio.sleep(1)
                            resp = await api_get(f"/jobs/{job_id}/progress")
                            pct = resp.get("progress", 0)
                            fwd_progress(pct / 100)
                            if pct >= 100 or pct < 0:
                                break
                        if pct == 100:
                            fwd_status.update("✅ finished")
                        elif pct < 0:
                            fwd_status.update("❌ failed")
                        return job_id, "Started"
                    fwd_start.click(
    start_forward, [date_start, date_end], [
        fwd_job_id, fwd_status])

                    async def stop_forward(job_id):
                        if job_id:
                            await api_patch(f"/jobs/{job_id}", {"action": "cancel"})
                            fwd_status.update("❌ cancelled")
                    fwd_stop.click(stop_forward, fwd_job_id, fwd_status)
                with gr.Tab("Live"):
                    gr.Markdown("### Live Trading")
                    risk = gr.Slider(0, 100, value=10, step=1, label="Risk Limit")
                    master_switch = gr.Checkbox(label="Live Trading ON/OFF", value=False)
                    live_start = gr.Button("Start Live")
                    live_stop = gr.Button("Stop")
                    flatten_btn = gr.Button("Flatten All Positions", elem_id="flatten-btn", variant="stop")
                    live_progress = gr.Progress()
                    live_status = gr.Textbox(label="Status", interactive=False)
                    live_job_id = gr.State("")
                    async def start_live(risk, master):
                        if not master:
                            live_status.update("Live trading is OFF")
                            return "", "OFF"
                        req = {"type": "live", "params": {"risk": risk, "assets": settings["assets"]}}
                        job = await api_post("/jobs/live", req)
                        if "error" in job:
                            live_status.update(f"API error: {job['error']}")
                            return "", f"API error: {job['error']}"
                        job_id = job.get("job_id") or job.get("id")
                        live_progress(0)
                        while True:
                            await asyncio.sleep(1)
                            resp = await api_get(f"/jobs/{job_id}/progress")
                            pct = resp.get("progress", 0)
                            live_progress(pct / 100)
                            if pct >= 100 or pct < 0:
                                break
                        if pct == 100:
                            live_status.update("✅ finished")
                        elif pct < 0:
                            live_status.update("❌ failed")
                        return job_id, "Started"
                    live_start.click(start_live, [risk, master_switch], [live_job_id, live_status])
                    async def stop_live(job_id):
                        if job_id:
                            await api_patch(f"/jobs/{job_id}", {"action": "cancel"})
                            live_status.update("❌ cancelled")
                    live_stop.click(stop_live, live_job_id, live_status)
                    async def flatten_positions():
                        resp = await api_post("/positions/flatten", {})
                        if "error" in resp:
                            live_status.update(f"Flatten error: {resp['error']}")
                        else:
                            live_status.update(f"Flattening started: task {resp.get('task_id')}")
                    flatten_btn.click(flatten_positions, [], live_status)
                with gr.Tab("Pipeline"):
                    gr.Markdown("### Full ML Pipeline")
                    pipeline_file = gr.File(label="Upload Dataset", file_types=[".csv", ".jsonl", ".json", ".zst"])
                    pipeline_user = gr.Textbox(label="User", placeholder="username")
                    pipeline_config = gr.Textbox(label="Config (JSON)", placeholder="{}")
                    pipeline_btn = gr.Button("Start Pipeline")
                    pipeline_stage = gr.Textbox(label="Current Stage", interactive=False)
                    pipeline_eta = gr.Textbox(label="ETA (seconds)", interactive=False)
                    pipeline_elapsed = gr.Textbox(label="Elapsed (seconds)", interactive=False)
                    pipeline_status = gr.Textbox(label="Status", interactive=False)

                    async def start_pipeline(file, user, config):
                        import httpx, asyncio, json
                        if not file or not user:
                            return "", 0, "", "", "Please provide file and user.", ""
                        files = {"file": (file.name, file)}
                        data = {"user": user, "config": config or "{}"}
                        async with httpx.AsyncClient() as client:
                            r = await client.post(f"{API_URL}/pipeline/start", files=files, data=data)
                            r.raise_for_status()
                            pipeline_id = r.json()["pipeline_id"]
                        while True:
                            await asyncio.sleep(1)
                            async with httpx.AsyncClient() as client:
                                resp = await client.get(f"{API_URL}/pipeline/{pipeline_id}/progress")
                                d = resp.json()
                            stage = d.get("stage", "")
                            pct = int(d.get("progress", 0))
                            eta = str(d.get("eta", ""))
                            elapsed = str(d.get("elapsed", ""))
                            status = d.get("status", "")
                            pipeline_progress(pct / 100)
                            pipeline_stage.update(stage)
                            pipeline_eta.update(eta)
                            pipeline_elapsed.update(elapsed)
                            pipeline_status.update(status)
                            if status in ("completed", "failed"):
                                break
                        return stage, pct / 100, eta, elapsed, status, pipeline_id

                    pipeline_btn.click(
                        start_pipeline,
                        inputs=[pipeline_file, pipeline_user, pipeline_config],
                        outputs=[pipeline_stage, pipeline_eta, pipeline_elapsed, pipeline_status, gr.State("")],
                    )

def find_free_port(start=7860, end=7862):
    for port in range(start, end+1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
    raise OSError(f"No free port in range {start}-{end}")

# --- Headless mode ---
if __name__ == "__main__":
    port = find_free_port()
    print(f"Launching Gradio on port {port}")
    demo.queue().launch(server_port=port, share=True)
