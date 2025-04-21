#!/bin/bash
set -e

# 1. Run Alembic migrations
cd /home/clearpeakegroup/finrl-platform/backend
if [ -f alembic.ini ]; then
    alembic upgrade head
fi

# 2. Start Docker Compose services (includes backend, db, redis, celery worker)
cd /home/clearpeakegroup/finrl-platform/infra
if command -v docker-compose &> /dev/null; then
    docker-compose up -d
else
    docker compose up -d
fi

# 3. Build React frontend for production
cd /home/clearpeakegroup/finrl-platform/frontend-ui
if [ -f package.json ]; then
    npm install
    npm run build
    # Install serve if not present
    if ! command -v serve &> /dev/null; then
        npm install -g serve
    fi
    # Serve the build in the background
    nohup serve -s dist -l 5173 &
fi

# 4. Start backend in a new terminal (if not using Docker Compose for backend)
cd /home/clearpeakegroup/finrl-platform/backend
nohup gnome-terminal -- bash -c "/usr/bin/python3 main.py; exec bash" &

# 5. Start Gradio frontend in a new terminal
cd /home/clearpeakegroup/finrl-platform/frontend
nohup gnome-terminal -- bash -c "/usr/bin/python3 app.py; exec bash" & 