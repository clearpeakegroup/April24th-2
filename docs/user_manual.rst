User Manual
===========

- How to upload data
- How to train an agent
- How to run backtests
- How to run forward tests
- How to go live
- Using the Gradio UI 

Gradio UI for RIG 3.0 Trading Workbench
======================================

To launch the Gradio UI:

.. code-block:: bash

   scripts/launch_gradio.sh

Features:

- Upload strategy config (YAML/JSON)
- Start training jobs and monitor progress
- Launch live trading bots and stream PnL

All actions are wired to backend services and require Redis and Celery to be running. 