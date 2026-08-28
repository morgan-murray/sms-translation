from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import ctranslate2
import sentencepiece as spm
import yaml


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


def safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(root):
                raise RuntimeError(f"unsafe path in {archive.name}: {member.filename}")
        bundle.extractall(destination)


def locate(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name} under {root}, found {len(matches)}")
    return matches[0]


def write_source_metadata(destination: Path, spec: dict) -> None:
    metadata = {
        "model_id": spec["model_id"],
        "revision": spec["revision"],
        "license": spec["license"],
        "quantization": "int8",
    }
    (destination / "MODEL_SOURCE.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def install_opus(spec: dict, staging_root: Path) -> None:
    final = MODEL_ROOT / "specialists" / spec["pair"]
    if ctranslate2.contains_model(str(final)) and (final / "source.spm").is_file() and (final / "target.spm").is_file():
        print(f"Verified installed {spec['pair']}", flush=True)
        return
    archive = staging_root / f"{spec['pair']}.zip"
    download(spec["url"], archive, size=spec["size"], checksum=spec["md5"], algorithm="md5")
    extracted = staging_root / f"{spec['pair']}-source"
    extracted.mkdir()
    safe_extract(archive, extracted)
    converted = staging_root / f"{spec['pair']}-converted"
    subprocess.run(
        [
            "ct2-opus-mt-converter",
            "--model_dir",
            str(extracted),
            "--output_dir",
            str(converted),
            "--quantization",
            "int8",
        ],
        check=True,
    )
    shutil.copy2(locate(extracted, "source.spm"), converted / "source.spm")
    shutil.copy2(locate(extracted, "target.spm"), converted / "target.spm")
    write_source_metadata(converted, spec)
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(final, ignore_errors=True)
    converted.replace(final)
    archive.unlink(missing_ok=True)
    shutil.rmtree(extracted, ignore_errors=True)
    print(f"Installed {spec['pair']}", flush=True)


def sentencepiece_vocab(model_path: Path, output_path: Path) -> None:
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    vocabulary = {processor.id_to_piece(index): index for index in range(processor.get_piece_size())}
    output_path.write_text(
        yaml.safe_dump(vocabulary, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def install_hplt(spec: dict, staging_root: Path) -> None:
    final = MODEL_ROOT / "specialists" / spec["pair"]
    if ctranslate2.contains_model(str(final)) and (final / "source.spm").is_file() and (final / "target.spm").is_file():
        print(f"Verified installed {spec['pair']}", flush=True)
        return
    source = staging_root / f"{spec['pair']}-source"
    source.mkdir()
    for file_spec in spec["files"]:
        url = (
            f"https://huggingface.co/{spec['model_id']}/resolve/{spec['revision']}/"
            f"{file_spec['name']}?download=true"
        )
        download(
            url,
            source / file_spec["name"],
            size=file_spec["size"],
            checksum=file_spec["sha256"],
            algorithm="sha256",
        )
    spm_path = next(source.glob("*.spm"))
    vocab_path = source / "vocab.yml"
    sentencepiece_vocab(spm_path, vocab_path)
    converted = staging_root / f"{spec['pair']}-converted"
    subprocess.run(
        [
            "ct2-marian-converter",
            "--model_path",
            str(source / "model.npz.best-chrf.npz"),
            "--vocab_paths",
            str(vocab_path),
            str(vocab_path),
            "--output_dir",
            str(converted),
            "--quantization",
            "int8",
        ],
        check=True,
    )
    shutil.copy2(spm_path, converted / "source.spm")
    shutil.copy2(spm_path, converted / "target.spm")
    write_source_metadata(converted, spec)
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(final, ignore_errors=True)
    converted.replace(final)
    shutil.rmtree(source, ignore_errors=True)
    print(f"Installed {spec['pair']}", flush=True)


def install_gguf(spec: dict) -> None:
    destination = MODEL_ROOT / "gguf" / spec["name"]
    url = f"https://huggingface.co/{spec['model_id']}/resolve/{spec['revision']}/{spec['name']}?download=true"
    download(url, destination, size=spec["size"], checksum=spec["sha256"], algorithm="sha256")


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="translation-models-", dir=MODEL_ROOT) as staging:
        staging_root = Path(staging)
        for spec in manifest["specialists"]:
            if spec["kind"] == "opus_zip":
                install_opus(spec, staging_root)
            else:
                install_hplt(spec, staging_root)
        for spec in manifest["gguf"]:
            install_gguf(spec)
    print("All model artifacts are installed and verified.", flush=True)


if __name__ == "__main__":
    main()
