import base64, json, subprocess, sys
from pathlib import Path

REPO   = "tomas-fryza/iaeste26-blasting-sound"
NB_DIR = Path("notebooks/validations")

def gh(method, endpoint, body=None, allow_404=False):
    cmd = ["gh","api","--method",method,f"/repos/{REPO}/{endpoint}",
           "--header","Accept: application/vnd.github+json"]
    if body:
        bf = Path("_gh_body.json")
        bf.write_text(json.dumps(body), encoding="utf-8")
        cmd += ["--input", str(bf)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        if allow_404 and "404" in r.stderr: return None
        print("ERROR:", r.stderr[:300]); return None
    return json.loads(r.stdout) if r.stdout.strip() else {}

notebooks = sorted(NB_DIR.glob("validation_*.ipynb"))
print(f"Pushing {len(notebooks)} notebooks to GitHub...")

for nb in notebooks:
    remote = f"notebooks/validations/{nb.name}"
    info   = gh("GET", f"contents/{remote}", allow_404=True)
    sha    = info.get("sha") if info else None
    b64    = base64.b64encode(nb.read_bytes()).decode("ascii")
    body   = {"message": f"Fix and execute {nb.name} (rglob dedup, figure-close, sample-size cap; validation_06 also fixes an elliptical-vs-bessel stopband test that was sampling the wrong frequency point)", "content": b64}
    if sha: body["sha"] = sha
    result = gh("PUT", f"contents/{remote}", body=body)
    new_sha = (result or {}).get("content", {}).get("sha", "?")
    print(f"  {'UPDATE' if sha else 'NEW   '} {nb.name} → {new_sha[:7]}")

print("Done.")
