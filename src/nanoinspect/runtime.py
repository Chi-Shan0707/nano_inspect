"""ModelScope acquisition, strict offline loading, baseline, and provenance."""
from __future__ import annotations
import contextlib
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parents[2]


def read_config(config: str | Path | dict | None = None) -> dict:
    with open(ROOT / "configs/model.yaml") as f:
        defaults = yaml.safe_load(f)
    if isinstance(config, dict):
        return {**defaults, **config}
    with open(config or ROOT / "configs/model.yaml") as f:
        return {**defaults, **yaml.safe_load(f)}


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=lambda x: sorted(x) if isinstance(x, set) else str(x)) + "\n")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict:
    names = ["torch", "transformers", "modelscope", "safetensors", "tokenizers", "numpy", "PyYAML"]
    return {name: importlib.metadata.version(name) for name in names}


def doctor() -> dict:
    import torch
    return {"python": platform.python_version(), "platform": platform.platform(),
            "environment": os.environ.get("CONDA_DEFAULT_ENV"), "python_executable": __import__("sys").executable,
            "packages": package_versions(), "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
            "gpu_memory": list(torch.cuda.mem_get_info()) if torch.cuda.is_available() else None,
            "cpu_count": os.cpu_count(), "memory": Path("/proc/meminfo").read_text(),
            "nvidia_smi": subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout if shutil.which("nvidia-smi") else None}


def download(config=None) -> dict:
    from modelscope.hub.api import HubApi
    from modelscope import snapshot_download
    cfg = read_config(config)
    model_dir = (ROOT / cfg["model_dir"]).resolve()
    api = HubApi()
    requested = cfg.get("revision", "master")
    remote_url = f"https://www.modelscope.cn/{cfg['model_id']}.git"
    ref = subprocess.check_output(["git", "ls-remote", remote_url, f"refs/heads/{requested}", f"refs/tags/{requested}"], text=True).strip()
    resolved = ref.split()[0] if ref else requested
    files = api._api.legacy.list_repo_files(cfg["model_id"], "model", revision=resolved, recursive=True)
    # The API's per-file Revision is an immutable git object identifier.
    revisions = sorted({f.get("Revision") for f in files if f.get("Revision")})
    revision_detail = api.get_valid_revision_detail(cfg["model_id"], requested) if len(requested) != 40 else {"Revision": requested}
    snapshot_download(cfg["model_id"], revision=resolved, local_dir=str(model_dir))
    expected_paths = {f["Path"] for f in files if f.get("Type") != "tree"}
    artifacts = [{"path": str(p.relative_to(model_dir)), "bytes": p.stat().st_size, "sha256": sha256(p)}
                 for p in sorted(model_dir.rglob("*")) if p.is_file() and str(p.relative_to(model_dir)) in expected_paths]
    hashes = {x["path"]: x["sha256"] for x in artifacts}
    for entry in files:
        if entry.get("Type") == "tree": continue
        if entry["Path"] not in hashes:
            raise RuntimeError(f"Complete snapshot missing {entry['Path']}")
        if entry.get("Sha256") and hashes[entry["Path"]] != entry["Sha256"]:
            raise RuntimeError(f"Upstream hash mismatch: {entry['Path']}")
    import requests
    version = importlib.metadata.version("transformers")
    ref = requests.get(f"https://api.github.com/repos/huggingface/transformers/git/ref/tags/v{version}", timeout=30)
    ref.raise_for_status()
    source_revision = ref.json()["object"]["sha"]
    source_url = f"https://raw.githubusercontent.com/huggingface/transformers/{source_revision}/src/transformers/models/qwen3_5/modeling_qwen3_5.py"
    source_response = requests.get(source_url, timeout=30)
    source_response.raise_for_status()
    upstream_source_sha256 = hashlib.sha256(source_response.content).hexdigest()
    source_path = importlib.metadata.distribution("transformers").locate_file("transformers/models/qwen3_5/modeling_qwen3_5.py")
    if sha256(source_path) != upstream_source_sha256:
        raise RuntimeError("Installed Transformers model source does not match pinned upstream commit")
    result = {"provider": "ModelScope", "model_id": cfg["model_id"], "requested_revision": requested,
              "resolved_file_revisions": revisions, "revision_detail": revision_detail, "download_revision": resolved,
              "revision_limit": "ModelScope API exposes branch and file hashes; immutable repository commit unavailable" if resolved == requested else None,
              "transformers_source_revision": source_revision, "upstream_hashes_verified": True,
              "upstream_transformers_source_sha256": upstream_source_sha256,
              "model_dir": str(model_dir), "api_files": files,
              "artifacts": artifacts, "acquisition_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()}
    write_json(model_dir / "nanoinspect_provenance.json", result)
    return result


@dataclass
class Runtime:
    model: Any
    tokenizer: Any
    device: Any
    dtype: Any
    model_dir: Path
    provenance: dict
    config: dict
    loading_info: dict


def load_model(config=None) -> Runtime:
    # Set before importing Transformers; all loads additionally require local_files_only.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    cfg = read_config(config)
    model_dir = (ROOT / cfg["model_dir"]).resolve()
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"Local snapshot absent: {model_dir}. Run nanoinspect download first.")
    torch.manual_seed(cfg.get("seed", 0))
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    dtype_name = cfg.get("dtype", "bfloat16")
    if device.type == "cuda" and dtype_name == "bfloat16" and not torch.cuda.is_bf16_supported():
        dtype_name = "float16"
    if device.type == "cpu" and dtype_name == "float16":
        dtype_name = "float32"
    dtype = getattr(torch, dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    model, info = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False, dtype=dtype,
        attn_implementation=cfg.get("attn_implementation", "eager"), output_loading_info=True)
    model.to(device).eval()
    import transformers.models.qwen3_5.modeling_qwen3_5 as implementation
    source = Path(inspect.getfile(implementation))
    provenance_path = model_dir / "nanoinspect_provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    provenance = {**provenance, "packages": package_versions(), "transformers_source_path": str(source),
                  "transformers_source_sha256": sha256(source),
                  "transformers_immutable_source_url": f"https://github.com/huggingface/transformers/blob/{provenance.get('transformers_source_revision', 'v' + transformers.__version__)}/src/transformers/models/qwen3_5/modeling_qwen3_5.py",
                  "device": str(device), "dtype": str(dtype), "local_files_only": True,
                  "loading_info": info}
    info = json.loads(json.dumps(info, default=lambda x: sorted(x) if isinstance(x, set) else str(x)))
    provenance["loading_info"] = info
    return Runtime(model, tokenizer, device, dtype, model_dir, provenance, cfg, info)


@contextlib.contextmanager
def deny_network():
    """Prove a second load/inference cannot initiate network connections."""
    original_connect = socket.socket.connect
    original_create = socket.create_connection
    attempts = []
    def denied(*args, **kwargs):
        attempts.append(str(args[1:])); raise RuntimeError("Network disabled for offline verification")
    socket.socket.connect = denied
    socket.create_connection = denied
    try:
        yield attempts
    finally:
        socket.socket.connect = original_connect
        socket.create_connection = original_create


def baseline(config=None, output="runs/baseline", offline_check=False) -> dict:
    import torch
    out_dir = Path(output)
    context = deny_network() if offline_check else contextlib.nullcontext([])
    with context as attempts:
        rt = load_model(config)
        cfg = rt.config
        messages = [{"role": "user", "content": cfg["prompt"]}]
        formatted = rt.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                     enable_thinking=False)
        inputs = rt.tokenizer(formatted, return_tensors="pt").to(rt.device)
        if rt.device.type == "cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            result = rt.model(**inputs, use_cache=False)
            positions = sorted(set([0, inputs.input_ids.shape[1] // 2, inputs.input_ids.shape[1] - 1]))
            selected = result.logits[0, positions].float().cpu()
            del result
            generated = rt.model.generate(**inputs, do_sample=False, max_new_tokens=cfg.get("max_new_tokens", 24),
                                          use_cache=True)
        if rt.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        values, ids = selected.topk(10, dim=-1)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"positions": positions, "logits": selected}, out_dir / "selected_logits.pt")
        report = {"kind": "baseline", "prompt": cfg["prompt"], "formatted_prompt": formatted,
                  "input_ids": inputs.input_ids.tolist(), "selected_positions": positions,
                  "selected_logits_file": "selected_logits.pt", "top10_ids": ids.tolist(), "top10_logits": values.tolist(),
                  "generated_ids": generated.tolist(), "continuation_ids": generated[:, inputs.input_ids.shape[1]:].tolist(),
                  "generation": rt.tokenizer.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True),
                  "elapsed_seconds": elapsed, "max_new_tokens": cfg.get("max_new_tokens",24), "do_sample": False,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated() if rt.device.type == "cuda" else None,
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved() if rt.device.type == "cuda" else None,
                  "offline_network_blocked": offline_check, "network_attempts": attempts,
                  "chat_template_sha256": hashlib.sha256(rt.tokenizer.chat_template.encode()).hexdigest(),
                  "generation_config": rt.model.generation_config.to_dict(), "provenance": rt.provenance}
        write_json(out_dir / "baseline.json", report)
        return report


def compare_baselines(reference, candidate, output=None):
    """Check deterministic reproduction using saved full-vocabulary selected logits."""
    import torch
    reference, candidate = Path(reference), Path(candidate)
    a = json.loads((reference / "baseline.json").read_text())
    b = json.loads((candidate / "baseline.json").read_text())
    x = torch.load(reference / "selected_logits.pt", weights_only=True)["logits"]
    y = torch.load(candidate / "selected_logits.pt", weights_only=True)["logits"]
    report = {"kind": "baseline_comparison", "reference": str(reference), "candidate": str(candidate),
              "input_ids_equal": a["input_ids"] == b["input_ids"],
              "generated_ids_equal": a["generated_ids"] == b["generated_ids"],
              "logits_exact_equal": torch.equal(x, y),
              "max_absolute_logit_difference": (x - y).abs().max().item(),
              "offline_network_blocked": b["offline_network_blocked"], "network_attempts": b["network_attempts"]}
    write_json(output or candidate / "comparison.json", report)
    if not all(report[k] for k in ("input_ids_equal", "generated_ids_equal", "logits_exact_equal")):
        raise RuntimeError(f"Baseline comparison failed; see {output or candidate / 'comparison.json'}")
    return report
