import psutil
import requests
import redis
import psycopg2
import os

def check_system():
    print(f"CPU: {psutil.cpu_percent()}%")
    print(f"Memory: {psutil.virtual_memory().percent}%")
    print(f"Disk: {psutil.disk_usage('/').percent}%")

def check_redis(url):
    try:
        r = redis.Redis.from_url(url)
        r.ping()
        print("Redis: OK")
    except Exception as e:
        print(f"Redis: FAIL ({e})")

def check_postgres(host, port, user, password, db):
    try:
        conn = psycopg2.connect(host=host, port=port, user=user, password=password, dbname=db)
        conn.close()
        print("Postgres: OK")
    except Exception as e:
        print(f"Postgres: FAIL ({e})")

def check_databento():
    # Stub: Replace with real API check
    try:
        requests.get("https://databento.com", timeout=3)
        print("Databento: OK")
    except Exception as e:
        print(f"Databento: FAIL ({e})")

def check_etrade():
    # Stub: Replace with real API check
    try:
        requests.get("https://apisb.etrade.com", timeout=3)
        print("E*TRADE: OK")
    except Exception as e:
        print(f"E*TRADE: FAIL ({e})")

if __name__ == "__main__":
    print("--- System Health Check ---")
    check_system()
    check_redis(os.getenv("REDIS_URL", "redis://localhost:6380/0"))
    check_postgres("localhost", 5433, "finrl", "finrlpass", "finrl_db")
    check_databento()
    check_etrade() 