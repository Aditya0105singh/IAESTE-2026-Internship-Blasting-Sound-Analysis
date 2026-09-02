import base64, json, subprocess, sys
from pathlib import Path

REPO   = "tomas-fryza/iaeste26-blasting-sound"
NB     = Path("notebooks/validation_report.ipynb")
REMOTE = "notebooks/validation_report.ipynb"
MSG    = "Add validation_report.ipynb — validates all 11 plots with PASS/FAIL checks"

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
        print("ERROR:", r.stderr[:500]); sys.exit(1)
    return json.loads(r.stdout) if r.stdout.strip() else {}

info = gh("GET", f"contents/{REMOTE}", allow_404=True)
sha  = info.get("sha") if info else None
print(f"Existing SHA: {sha or 'none (new file)'}")

content_b64 = base64.b64encode(NB.read_bytes()).decode("ascii")
print(f"File size: {NB.stat().st_size:,} bytes")

body = {"message": MSG, "content": content_b64}
if sha:
    body["sha"] = sha

result = gh("PUT", f"contents/{REMOTE}", body=body)
print(f"Pushed. New SHA: {result.get('content',{}).get('sha','?')}")
