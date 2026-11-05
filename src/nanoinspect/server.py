"""Localhost workbench: a small JSON API over ``workbench`` plus static pages.

Standard library only, bound to 127.0.0.1 by default. The model loads once in
a background thread; analyses run one at a time behind a lock because they
share one model and register temporary hooks on it.
"""
from __future__ import annotations

import json
import math
import mimetypes
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse
from . import workbench
from .runtime import ROOT

WEB = Path(__file__).with_name("web")
STATIC_ROOTS = {"/static/": WEB, "/runs/": ROOT / "runs", "/docs/": ROOT / "docs"}


class State:
    def __init__(self, config=None):
        self.config, self.runtime, self.error = config, None, None
        self.status, self.load_seconds = "idle", None
        self.load_lock, self.run_lock = threading.Lock(), threading.Lock()

    def load(self):
        with self.load_lock:
            if self.runtime is not None or self.status == "loading":
                return
            self.status, started = "loading", time.perf_counter()
        try:
            from .runtime import load_model
            runtime = load_model(self.config)
            with self.load_lock:
                self.runtime, self.status = runtime, "ready"
                self.load_seconds = round(time.perf_counter() - started, 2)
        except Exception as exc:  # surfaced through /api/status
            with self.load_lock:
                self.status, self.error = "error", f"{type(exc).__name__}: {exc}"

    def require(self):
        if self.runtime is None:
            if self.status == "idle":
                threading.Thread(target=self.load, daemon=True).start()
            raise RuntimeError(f"Model is {self.status}" + (f" ({self.error})" if self.error else "") + "; retry shortly")
        return self.runtime

    def describe(self):
        info = {"status": self.status, "error": self.error, "load_seconds": self.load_seconds}
        if self.runtime is not None:
            import torch
            rt = self.runtime
            info.update(model_id=rt.config.get("model_id"), revision=rt.provenance.get("download_revision"),
                        device=str(rt.device), dtype=str(rt.dtype).replace("torch.", ""),
                        layers=len(workbench.layer_types(rt.model)),
                        gpu=torch.cuda.get_device_name() if rt.device.type == "cuda" else None,
                        allocated_bytes=torch.cuda.memory_allocated() if rt.device.type == "cuda" else None)
        assets = workbench.steering_assets()
        info["steering"] = None if assets is None else {
            "layers": sorted(assets["directions"]), "selected_layer": assets["selected_layer"],
            "random": sorted(assets["random"])}
        return info


def _flag(args, key):
    return bool(args.get(key, False))


ANALYSES = {
    "lens": lambda rt, a: workbench.lens(rt, a["prompt"], _flag(a, "chat_template"), int(a.get("top_k", 5)),
                                         a.get("target") or None),
    "walkthrough": lambda rt, a: workbench.walkthrough(rt, a["prompt"], a["layer"], int(a.get("position", -1)),
                                                       a.get("head", "auto"), a.get("compare") or None,
                                                       _flag(a, "chat_template")),
    "mixer": lambda rt, a: workbench.mixer(rt, a["prompt"], a["layer"], _flag(a, "chat_template")),
    "patch": lambda rt, a: workbench.patch(rt, a["clean"], a["corrupted"], a.get("sites", ["residual_output"]),
                                           a.get("target") or None, a.get("foil") or None, _flag(a, "chat_template")),
    "steer": lambda rt, a: workbench.steer(rt, a["prompt"], a["layer"], a.get("alpha", 0.25), a.get("direction", "contrastive"),
                                           a.get("steps", 24), a.get("policy", "decode_time"), _flag(a, "chat_template"),
                                           a.get("positive"), a.get("negative")),
    "sweep": lambda rt, a: workbench.steer_sweep(rt, a["prompt"], a["layer"], a.get("alphas", [-0.5, -0.25, 0, 0.25, 0.5]),
                                                 a.get("direction", "contrastive"), a.get("control", "random_seed_0"),
                                                 a.get("steps", 16), a.get("policy", "decode_time"), _flag(a, "chat_template"),
                                                 a.get("positive"), a.get("negative"),
                                                 a.get("watch_positive"), a.get("watch_negative")),
}


def static_path(url_path):
    """Map a URL path onto a file under an allowed root, refusing traversal outside it."""
    for prefix, base in STATIC_ROOTS.items():
        if url_path.startswith(prefix):
            base = base.resolve()
            target = (base / unquote(url_path[len(prefix):])).resolve()
            if target.is_relative_to(base) and target.is_file():
                return target
            return None
    return None


def _finite(value):
    """Strict JSON: non-finite floats (e.g. masked attention scores) become the strings "inf", "-inf", "nan"."""
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        server_version = "nanoinspect"

        def log_message(self, fmt, *args):
            if not self.path.startswith(("/static/", "/api/status")):
                super().log_message(fmt, *args)

        def _json(self, value, code=200):
            body = json.dumps(_finite(value), default=str, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _file(self, path):
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path in ("/", "/index.html"):
                    return self._file(WEB / "index.html")
                if path == "/api/status":
                    return self._json(state.describe())
                if path == "/api/architecture":
                    return self._json(workbench.architecture())
                if path == "/api/runs":
                    return self._json(workbench.runs())
                target = static_path(path)
                if target is None:
                    return self._json({"error": "Not found"}, 404)
                return self._file(target)
            except Exception as exc:
                return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

        def do_POST(self):
            name = urlparse(self.path).path.removeprefix("/api/")
            if name not in ANALYSES:
                return self._json({"error": "Not found"}, 404)
            try:
                length = int(self.headers.get("Content-Length", 0))
                args = json.loads(self.rfile.read(length) or b"{}")
                runtime = state.require()
                with state.run_lock:
                    started = time.perf_counter()
                    result = ANALYSES[name](runtime, args)
                    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
                return self._json(result)
            except RuntimeError as exc:
                return self._json({"error": str(exc)}, 503 if "Model is" in str(exc) else 400)
            except (KeyError, ValueError, IndexError, TypeError) as exc:
                return self._json({"error": f"{type(exc).__name__}: {exc}"}, 400)
            except Exception as exc:
                traceback.print_exc()
                return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
    return Handler


def serve(host="127.0.0.1", port=8765, config=None, preload=True):
    state = State(config)
    if preload:
        threading.Thread(target=state.load, daemon=True).start()
    server = ThreadingHTTPServer((host, port), make_handler(state))
    print(f"nanoinspect workbench on http://{host}:{port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
