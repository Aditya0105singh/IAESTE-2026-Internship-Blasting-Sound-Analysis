"""Push notebook to GitHub via API (avoids downloading 6GB data repo)."""
import base64, json, subprocess, sys
from pathlib import Path

REPO    = "tomas-fryza/iaeste26-blasting-sound"
NB_PATH = Path("notebooks/blasting_analysis.ipynb")
REMOTE  = "notebooks/blasting_analysis.ipynb"
MSG     = "Fix: dedupe WAV file scan (was double-counting via case-insensitive rglob), close matplotlib figures per cell, cap multi-file loop to a 2-file sample with periodic gc.collect() to avoid OOM. Verified: 23/23 cells executed, 0 errors, 11 plots."

def gh(method, endpoint, body=None, allow_404=False):
    cmd = ["gh", "api", "--method", method, f"/repos/{REPO}/{endpoint}",
           "--header", "Accept: application/vnd.github+json"]
    if body:
        body_file = Path("_gh_body.json")
        body_file.write_text(json.dumps(body), encoding="utf-8")
        cmd += ["--input", str(body_file)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        if allow_404 and "404" in r.stderr:
            return None
        print("ERROR:", r.stderr[:500])
        sys.exit(1)
    return json.loads(r.stdout) if r.stdout.strip() else {}

# Get current SHA
info = gh("GET", f"contents/{REMOTE}", allow_404=True)
sha  = info.get("sha") if info else None
print(f"Current SHA: {sha}")

# Encode notebook
content_b64 = base64.b64encode(NB_PATH.read_bytes()).decode("ascii")
print(f"Notebook size: {len(NB_PATH.read_bytes()):,} bytes")

body = {"message": MSG, "content": content_b64}
if sha:
    body["sha"] = sha

result = gh("PUT", f"contents/{REMOTE}", body=body)
new_sha = result.get("content", {}).get("sha", "?")
print(f"Pushed successfully. New SHA: {new_sha}")
