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
        return {"error": str(e)}


async def api_post(path, data):
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(f"{API_URL}{path}", json=data)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        return {"error": str(e)}


async def api_patch(path, data):
    try:
        async with httpx.AsyncClient() as client:
            r = await client.patch(f"{API_URL}{path}", json=data)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        return {"error": str(e)}


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
                with gr.Tab("Dashboard"):
                    jobs_table = gr.Dataframe(headers=["ID", "Type", "Status", "% Complete"], datatype=["str", "str", "str", "number"], interactive=False, label="Jobs")
                    pl_chart = gr.LinePlot(x="Time", y="PnL", title="P/L Over Time", overlay_point=True, interactive=False)
                    jobs_data = []
                    pl_data = []
                    def update_jobs():
                        jobs = asyncio.run(api_get("/jobs"))
                        if isinstance(jobs, dict) and "error" in jobs:
                            return [["Error", jobs["error"], "", ""]]
                        return [[j["id"], j["type"], j["status"], j["percent_complete"]] for j in jobs]
                    def update_pl():
                        acc = asyncio.run(api_get("/account"))
                        if isinstance(acc, dict) and "error" in acc:
                            pl_data.append({"Time": len(pl_data), "PnL": 0})
                        else:
                            pl_data.append({"Time": len(pl_data), "PnL": acc["pnl"]})
                        return pl_data
                    jobs_table.value = update_jobs()
                    pl_chart.value = update_pl()
                    refresh_btn = gr.Button("Refresh Dashboard")
                    def refresh_dashboard():
                        jobs_table.value = update_jobs()
                        pl_chart.value = update_pl()
                    refresh_btn.click(refresh_dashboard, None, None)
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
                        job_id = job["id"]
                        async for msg in sse_stream(f"{API_URL}/jobs/{job_id}/stream"):
                            d = json.loads(msg)
                            progress(d.get("pct", 0) / 100)
                            if d.get("pct", 0) == 100:
                                status_box.update("✅ finished")
                                break
                            status_box.update(d.get("msg", ""))
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
                        job_id = job["id"]
                        async for msg in sse_stream(f"{API_URL}/jobs/{job_id}/stream"):
                            d = json.loads(msg)
                            fwd_progress(d.get("pct", 0) / 100)
                            if d.get("pct", 0) == 100:
                                fwd_status.update("✅ finished")
                                break
                            fwd_status.update(d.get("msg", ""))
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
                        job_id = job["id"]
                        async for msg in sse_stream(f"{API_URL}/jobs/{job_id}/stream"):
                            d = json.loads(msg)
                            live_progress(d.get("pct", 0) / 100)
                            if d.get("pct", 0) == 100:
                                live_status.update("✅ finished")
                                break
                            live_status.update(d.get("msg", ""))
                        return job_id, "Started"
                    live_start.click(start_live, [risk, master_switch], [live_job_id, live_status])
                    async def stop_live(job_id):
                        if job_id:
                            await api_patch(f"/jobs/{job_id}", {"action": "cancel"})
                            live_status.update("❌ cancelled")
                    live_stop.click(stop_live, live_job_id, live_status)

# --- Headless mode ---
if __name__ == "__main__":
    import sys
    if "--headless" in sys.argv:
        print("Running on http://localhost:7860")
        demo.queue().launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_api=False,
            inbrowser=False
        )
    else:
        demo.queue().launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_api=False,
            inbrowser=False
        )
