import json, uuid
from pathlib import Path

nb_path = Path("notebooks/blasting_analysis.ipynb")
with open(nb_path, encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if "id" not in cell:
        cell["id"] = str(uuid.uuid4())[:8]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Added IDs to {len(nb['cells'])} cells")
