from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path



MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "/models"))
MANIFEST_PATH = Path(os.environ.get("MODEL_MANIFEST", "/srv/translation/model-manifest.json"))


def digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download(url: str, destination: Path, *, size: int, checksum: str, algorithm: str) -> None:
    if destination.is_file() and destination.stat().st_size == size and digest(destination, algorithm) == checksum:
        print(f"Verified cached {destination.name}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    print(f"Downloading {destination.name} ({size / 1_000_000:.1f} MB)", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "leavebird-translation/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    if partial.stat().st_size != size:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"size mismatch for {destination.name}")
    actual = digest(partial, algorithm)
    if actual != checksum:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"{algorithm} mismatch for {destination.name}: {actual}")
    partial.replace(destination)


def install_gguf(spec: dict) -> None:
    destination = MODEL_ROOT / "gguf" / spec["name"]
    url = f"https://huggingface.co/{spec['model_id']}/resolve/{spec['revision']}/{spec['name']}?download=true"
    download(url, destination, size=spec["size"], checksum=spec["sha256"], algorithm="sha256")


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    for spec in manifest["gguf"]:
        install_gguf(spec)
    print("All model artifacts are installed and verified.", flush=True)


if __name__ == "__main__":
    main()
