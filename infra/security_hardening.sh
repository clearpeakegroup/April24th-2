#!/bin/bash
set -e

# 1. Configure UFW firewall
sudo ufw allow 8000/tcp   # FastAPI backend
sudo ufw allow 5173/tcp   # React frontend
sudo ufw allow 5432/tcp   # Postgres
sudo ufw allow 6379/tcp   # Redis
sudo ufw allow 3000/tcp   # Grafana (if used)
sudo ufw allow 9090/tcp   # Prometheus (if used)
sudo ufw --force enable

# 2. Set permissions on certs and .env files
if [ -d "../certs" ]; then
  sudo chown root:root ../certs/key.pem ../certs/cert.pem
  sudo chmod 600 ../certs/key.pem ../certs/cert.pem
fi
if [ -f "../.env" ]; then
  chmod 600 ../.env
fi

# 3. Print reminders for secrets management and user permissions
echo "[INFO] Ensure secrets (DB passwords, JWT secret, etc.) are stored in environment variables or .env files with 600 permissions."
echo "[INFO] Never commit secrets to version control."
echo "[INFO] Ensure only the intended user (clearpeakegroup) has access to code, data, and secrets."
echo "[INFO] For extra isolation, consider systemd ProtectSystem=full and ProtectHome=yes in your service files." 