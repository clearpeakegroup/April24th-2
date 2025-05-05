#!/usr/bin/env python3
import itertools
import subprocess
import hashlib
import os
import sys
import time
from pathlib import Path

# --- CONFIGURABLE VERSION MATRIX (edit as needed) ---
VERSION_MATRIX = {
    'finrl': ["0.3.6"],
    'ccxt': ["3.1.47", "3.0.0"],
    'gymnasium': ["0.29.1", "0.26.0"],
    'stable-baselines3': ["2.0.0", "1.7.0"],
    'torch': ["2.0.1", "1.13.1"],
    'numpy': ["1.24.3", "1.23.5"],
    'pandas': ["2.0.3", "1.5.3"],
    'scikit-learn': ["1.3.0", "1.2.2"],
    'matplotlib': ["3.7.2", "3.6.3"],
}

MAX_TRIES = 4000
LOG_FILE = Path("scripts/repair_attempts.log")
SUCCESS_FILE = Path("scripts/working_requirements.txt")
SUMMARY_FILE = Path("scripts/repair_summary.log")
TEMP_REQ = Path("/tmp/finrl_repair_requirements.txt")

# --- UTILS ---
def combo_hash(combo):
    s = ",".join(f"{k}={v}" for k, v in combo.items())
    return hashlib.sha256(s.encode()).hexdigest()[:16]

def log_attempt(combo, status, msg):
    with LOG_FILE.open("a") as f:
        f.write(f"{combo_hash(combo)} | {status} | {combo} | {msg}\n")

def already_tried(combo):
    if not LOG_FILE.exists():
        return False
    h = combo_hash(combo)
    with LOG_FILE.open() as f:
        for line in f:
            if line.startswith(h):
                return True
    return False

def write_requirements(combo):
    with TEMP_REQ.open("w") as f:
        for k, v in combo.items():
            f.write(f"{k}=={v}\n")

def cleanup_docker():
    subprocess.run(["docker", "system", "prune", "-f"], check=False)
    subprocess.run(["docker", "builder", "prune", "-f"], check=False)
    subprocess.run(["docker", "compose", "-f", "infra/docker-compose.yml", "down"], check=False)

def build_and_install():
    # Copy temp requirements to scripts/install_finrl.sh
    install_script = Path("scripts/install_finrl.sh")
    with install_script.open("w") as f:
        f.write("#!/bin/bash\nset -e\n")
        f.write("pip install --no-cache-dir -r /tmp/finrl_repair_requirements.txt\n")
        f.write("python -c 'import finrl; import ccxt; print(\"✅ FinRL installation validated\")'\n")
    install_script.chmod(0o755)
    # Build docker
    result = subprocess.run(["docker", "compose", "-f", "infra/docker-compose.yml", "build", "--no-cache"], capture_output=True, text=True)
    return result

def run_smoke_test():
    result = subprocess.run(["bash", "scripts/smoke_test.sh"], capture_output=True, text=True)
    return result

def main():
    combos = list(itertools.islice(itertools.product(*VERSION_MATRIX.values()), MAX_TRIES))
    keys = list(VERSION_MATRIX.keys())
    tried = 0
    successes = 0
    failures = 0
    for values in combos:
        combo = dict(zip(keys, values))
        if already_tried(combo):
            continue
        tried += 1
        print(f"[TRY {tried}] Testing combo: {combo}")
        write_requirements(combo)
        cleanup_docker()
        build_result = build_and_install()
        if build_result.returncode != 0:
            log_attempt(combo, "BUILD_FAIL", build_result.stderr[:500])
            failures += 1
            continue
        smoke_result = run_smoke_test()
        if smoke_result.returncode == 0:
            log_attempt(combo, "SUCCESS", "All smoke tests passed.")
            print(f"[SUCCESS] Working combo found: {combo}")
            with SUCCESS_FILE.open("w") as f:
                for k, v in combo.items():
                    f.write(f"{k}=={v}\n")
            break
        else:
            log_attempt(combo, "SMOKE_FAIL", smoke_result.stderr[:500])
            failures += 1
    # Summary
    with SUMMARY_FILE.open("w") as f:
        f.write(f"Tried: {tried}\nSuccesses: {successes}\nFailures: {failures}\n")
    print(f"[DONE] Tried {tried} combinations. Successes: {successes}, Failures: {failures}.")
    if not SUCCESS_FILE.exists():
        print("[REPORT] No working combination found. See scripts/repair_attempts.log for details.")

if __name__ == "__main__":
    main() 