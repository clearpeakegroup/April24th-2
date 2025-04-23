#!/usr/bin/env bash
set -euo pipefail
COMPOSE="docker compose -f infra/docker-compose.yml"

echo "🔧 1) Patch frontend/Dockerfile CMD …"
DOCKERFILE=frontend/Dockerfile
grep -q 'CMD \["python", "gradio_app.py", "--headless"\]' "$DOCKERFILE" || \
  echo 'CMD ["python", "gradio_app.py", "--headless"]' >> "$DOCKERFILE"

echo "🔧 2) Patch infra/Dockerfile duplicate else/fi …"
sed -Ei '/RUN .*pyproject\.toml/,/COPY \./c\
RUN if [ -f pyproject.toml ]; then \
        pip install --no-cache-dir poetry && poetry install --no-root ; \
    else \
        pip install --no-cache-dir -r requirements.txt ; \
    fi\nCOPY . .' infra/Dockerfile

echo "🔧 3) Pin wheels …"
REQ=frontend/requirements.txt
mkdir -p frontend
grep -q '^gradio=='        "$REQ" 2>/dev/null && \
  sed -i 's/^gradio==.*/gradio==5.24.0/' "$REQ" || echo 'gradio==5.24.0' >> "$REQ"
grep -q '^gradio_client==' "$REQ" 2>/dev/null && \
  sed -i 's/^gradio_client==.*/gradio_client==0.16.0/' "$REQ" || echo 'gradio_client==0.16.0' >> "$REQ"

echo "🔧 4) Ensure launch() uses ssr_mode=False block=True …"
python - <<'PY'
import pathlib, re, sys
p = pathlib.Path("frontend/gradio_app.py")
if not p.exists():
    sys.exit("frontend/gradio_app.py missing")
txt = p.read_text()
m = re.search(r"demo\.queue\(\)\.launch\((?P<args>[^)]*)\)", txt, re.S)
if not m:
    sys.exit("launch() call not found")
args = m.group("args")
if "ssr_mode" not in args:
    args += ", ssr_mode=False"
if "block=" not in args:
    args += ", block=True"
txt = re.sub(r"demo\.queue\(\)\.launch\([^)]*\)", f"demo.queue().launch({args})", txt, 1, re.S)
p.write_text(txt)
PY

echo "🔨 5) Build api + gradio images …"
$COMPOSE build api gradio

echo "🚀 6) Up containers …"
$COMPOSE up -d api gradio

echo "🔍 7) Show wheel versions …"
$COMPOSE exec gradio python - <<'PY'
import importlib.metadata as im, json, sys, os
print(json.dumps({"gradio": im.version("gradio"),
                  "gradio_client": im.version("gradio_client")}, indent=2))
PY

echo "🟢  Dashboard → http://localhost:7860  (hard-refresh with Ctrl-Shift-R)"
