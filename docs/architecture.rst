Architecture
============

.. image:: architecture_diagram.png
   :alt: Platform Architecture

- Microservices: backend, frontend, middleware, monitoring
- Message queues, Redis, Postgres, Celery, API Gateway
- Security, monitoring, and deployment pipelines 

Gradio Front-End and Backend Service Contract
============================================

- The Gradio UI (frontend/gradio_app.py) exposes tabs for config upload, training, and live trading.
- All user actions are routed to backend/services/gradio_hooks.py, which calls Celery tasks and updates Redis for progress.
- Celery workers execute training and live trading jobs, pushing progress to Redis for real-time UI updates.
- DB CRUD and schema modules (backend/db) handle config persistence. 