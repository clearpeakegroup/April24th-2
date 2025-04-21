#!/usr/bin/env bash
set -euo pipefail
FILE="frontend/gradio_app.py"

# ── new click‑handler block ───────────────────────────────────────────
read -r -d '' NEW_CLICK <<'PY'
# ─────────────────────────────────────────────────────────────────────
INPUTS  = [train_epochs, batch_size, learning_rate]        # tweak ↴
OUTPUTS = [status_box]                                     # tweak ↴

train_btn.click(
    fn=train_model,
    inputs=INPUTS,
    outputs=OUTPUTS,
    api_name="train_model",
    progress=gr.Progress(track_tqdm=False)  # Gradio 4 style
)
# ─────────────────────────────────────────────────────────────────────
PY
# ─────────────────────────────────────────────────────────────────────

python - <<PY
import pathlib, re, textwrap, sys
p = pathlib.Path(FILE)
src = p.read_text()
src = re.sub(
    r"train_btn\.click\((?:.|\n)*?\)\s*",
    textwrap.dedent(NEW_CLICK + "\n"),
    src,
    flags=re.M,
)
p.write_text(src)
print("➜ patched", p)
PY

# Remove any old gr.Progress lines that are now orphaned
sed -Ei '/gr\.Progress\(.*\)/d' "$FILE"

# Re‑build only the gradio image and restart its container
docker compose -f infra/docker-compose.yml build gradio
docker compose -f infra/docker-compose.yml up -d gradio

echo -e "\n── tailing gradio logs ────────────────────────────────\n"
docker logs -f --tail 30 infra-gradio-1
