"""Run from the repo root:  python make_space.py
Builds ./hf_space/ - a ready-to-upload folder for the Hugging Face Space."""
import shutil
from pathlib import Path

root = Path(__file__).resolve().parent
out = root / "hf_space"
if out.exists():
    shutil.rmtree(out)
out.mkdir()

for name in ("app.py", "requirements.txt", "README.md"):
    shutil.copy(root / "gradio_space" / name, out / name)
shutil.copy(root / "inference.py", out / "inference.py")
shutil.copytree(root / "models", out / "models")
print("Created", out)
for p in sorted(out.rglob("*")):
    if p.is_file():
        print("  ", p.relative_to(out), f"({p.stat().st_size:,} bytes)")
