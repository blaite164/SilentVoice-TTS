"""Small helpers for launching user-selected local LLMs on Windows."""

from __future__ import annotations

import atexit
import json
import os
import platform
import shutil
import socket
import subprocess
import threading
import time
from urllib.parse import urlsplit
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from core.config import DATA_DIR
from core.i18n import tr
from core.win_api import assign_process_to_kill_on_close_job, has_nvidia_gpu

LLAMA_DIR = DATA_DIR / "engines" / "llama.cpp"
LLAMA_EXE = LLAMA_DIR / "llama-server.exe"
LLAMA_MODEL_DIR = LLAMA_DIR / "models"
LLAMA_ARCHIVE_URL = "https://github.com/ggml-org/llama.cpp/releases/download/b12541/llama-b12541-bin-win-vulkan-x64.zip"

_PROCESS_LOCK = threading.RLock()
_LLAMA_PROCESSES: Dict[str, Dict[str, Any]] = {}


def _hidden_process_options() -> Dict[str, Any]:
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {}


def find_lms_cli() -> Optional[str]:
    for name in ("lms.exe", "lms"):
        found = shutil.which(name)
        if found:
            return found
    user_home = Path.home()
    candidates = (
        user_home / ".lmstudio" / "bin" / "lms.exe",
        user_home / ".lmstudio" / "bin" / "lms",
        Path(os.environ.get("LOCALAPPDATA", "")) / "LM Studio" / "bin" / "lms.exe",
    )
    return next((str(path) for path in candidates if path.is_file()), None)


def _run_lms(args: List[str], timeout: float = 60.0) -> subprocess.CompletedProcess:
    cli = find_lms_cli()
    if not cli:
        raise RuntimeError("LM Studio command-line tools were not found. Open LM Studio once and enable/install its CLI, then try again.")
    try:
        return subprocess.run(
            [cli, *args], capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace", **_hidden_process_options(),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("LM Studio took too long to respond. Please try again.") from exc


def list_lmstudio_models() -> List[Dict[str, str]]:
    result = _run_lms(["ls", "--llm", "--json"])
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "Could not list LM Studio models.").strip()[-1200:])
    output = (result.stdout or "").strip()
    # Some CLI versions print a short notice before the JSON payload.
    start_candidates = [p for p in (output.find("["), output.find("{")) if p >= 0]
    if not start_candidates:
        raise RuntimeError("LM Studio returned an unreadable model list.")
    data: Any = None
    for start in sorted(start_candidates):
        try:
            data = json.loads(output[start:])
            break
        except Exception:
            continue
    if isinstance(data, dict):
        data = data.get("models", data.get("data", []))
    if not isinstance(data, list):
        raise RuntimeError("LM Studio returned an unexpected model list.")
    models: List[Dict[str, str]] = []
    for item in data:
        if isinstance(item, str):
            key = item.strip()
            label = key
        elif isinstance(item, dict):
            if str(item.get("type", "llm")).lower() not in ("llm", "model", ""):
                continue
            path = str(item.get("path") or "").strip()
            key = str(item.get("modelKey") or item.get("model_key") or item.get("key") or path or "").strip()
            label = str(item.get("displayName") or item.get("name") or key).strip()
        else:
            continue
        if key:
            entry: Dict[str, str] = {"key": key, "label": label or key}
            if isinstance(item, dict) and path:
                entry["path"] = path
            models.append(entry)
    return models


def _import_lmstudio_staged_file(
    staged_file: Path,
    safe_repo: str,
    cancel_event: Optional[threading.Event] = None,
    timeout: float = 600.0,
) -> None:
    """Import a staged GGUF into LM Studio without blocking on CLI Y/n prompts."""
    cli = find_lms_cli()
    if not cli:
        raise RuntimeError("LM Studio command-line tools were not found. Open LM Studio once, then try again.")

    proc = subprocess.Popen(
        [cli, "import", str(staged_file), "--user-repo", safe_repo, "--yes"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        **_hidden_process_options(),
    )
    out_chunks: List[str] = []
    err_chunks: List[str] = []

    def _drain(pipe: Any, target: List[str]) -> None:
        if pipe is None:
            return
        try:
            data = pipe.read()
            if data:
                target.append(data)
        except Exception:
            pass
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    t_out = threading.Thread(target=_drain, args=(proc.stdout, out_chunks), daemon=True)
    t_err = threading.Thread(target=_drain, args=(proc.stderr, err_chunks), daemon=True)
    t_out.start()
    t_err.start()

    # lms import --yes suppresses warnings yet still prompts "? Do you wish to continue? (Y/n)".
    if proc.stdin is not None:
        try:
            proc.stdin.write("y\n" * 6)
            proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.stdin.close()
        except Exception:
            pass

    deadline = time.monotonic() + float(timeout)
    try:
        while proc.poll() is None:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Model download cancelled.")
            if time.monotonic() >= deadline:
                raise RuntimeError("LM Studio took too long to respond. Please try again.")
            time.sleep(0.2)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=3.0)
        except Exception:
            pass
        raise
    finally:
        t_out.join(timeout=2.0)
        t_err.join(timeout=2.0)

    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Model download cancelled.")
    if proc.returncode != 0:
        detail = ("".join(err_chunks) or "".join(out_chunks) or "").strip()
        raise RuntimeError(detail[-1000:] or "LM Studio could not import the downloaded model.")


def download_recommended_lmstudio_model(
    url: str,
    filename: str,
    model_repo: str,
    cancel_event: Optional[threading.Event] = None,
) -> str:
    """Resumably download a recommended GGUF, then import it into LM Studio."""
    safe_name = Path(str(filename or "")).name
    safe_repo = str(model_repo or "").strip().strip("/")
    if not safe_name.lower().endswith(".gguf") or safe_name != filename:
        raise RuntimeError("Invalid GGUF file name.")
    if len(safe_repo.split("/")) != 2 or any(part in (".", "..") for part in safe_repo.split("/")):
        raise RuntimeError("Invalid LM Studio model repository.")
    cli = find_lms_cli()
    if not cli:
        raise RuntimeError("LM Studio command-line tools were not found. Open LM Studio once, then try again.")

    staging_dir = DATA_DIR / "model_downloads" / "lmstudio"
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged_file = staging_dir / safe_name
    from core.tts_manager import tts_manager
    if not staged_file.is_file() or staged_file.stat().st_size == 0:
        tts_manager._download_url_with_progress(
            url, staged_file, "local_llm_download", 0, 100, safe_name,
            cancel_event=cancel_event,
        )
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Model download cancelled.")

    tts_manager.download_progress.emit(
        "local_llm_download",
        99,
        tr("ui_local_models_lms_importing", "Додаю модель у бібліотеку LM Studio…"),
    )
    _import_lmstudio_staged_file(staged_file, safe_repo, cancel_event=cancel_event, timeout=600.0)
    staged_file.unlink(missing_ok=True)
    tts_manager.download_progress.emit(
        "local_llm_download",
        100,
        tr("ui_local_models_lms_download_done", "Модель завантажено в LM Studio. Оновлюю список…"),
    )
    return safe_repo


def download_recommended_gguf(
    url: str,
    filename: str,
    cancel_event: Optional[threading.Event] = None,
) -> Path:
    """Download a known GGUF file into SilentVoice's llama.cpp model folder."""
    safe_name = Path(str(filename or "")).name
    if not safe_name.lower().endswith(".gguf") or safe_name != filename:
        raise RuntimeError("Invalid GGUF file name.")
    LLAMA_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    destination = LLAMA_MODEL_DIR / safe_name
    from core.tts_manager import tts_manager
    tts_manager._download_url_with_progress(
        url, destination, "local_llama_cpp", 0, 100, safe_name,
        cancel_event=cancel_event,
    )
    tts_manager.download_progress.emit(
        "local_llama_cpp",
        100,
        tr("ui_local_models_gguf_download_done", "Модель завантажено в SilentVoice. Тепер натисніть «Запустити та підключити»."),
    )
    return destination


def _models_at(base_url: str) -> List[str]:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/models", timeout=2.5)
        if response.status_code != 200:
            return []
        data = response.json()
        return [str(item.get("id", "")).strip() for item in data.get("data", []) if isinstance(item, dict) and item.get("id")]
    except Exception:
        return []


def start_lmstudio_model(model_key: str, progress=None) -> Tuple[str, List[str]]:
    model_key = str(model_key or "").strip()
    if not model_key:
        raise RuntimeError("Choose a model in the list first.")
    base_url = "http://127.0.0.1:1234/v1"
    if progress:
        progress(tr("ui_local_models_lms_server_starting", "Starting the LM Studio local server…"))
    # If it is already serving the chosen model, do not reload it.
    if model_key not in _models_at(base_url):
        server_result = _run_lms(["server", "start", "--port", "1234"], timeout=30)
        # A nonzero result is acceptable when the server is already running;
        # loading the model and the final API check below decide readiness.
        if progress:
            progress(tr("ui_local_models_lms_model_loading", "Loading the selected model in LM Studio…"))
        load_result = _run_lms(["load", model_key, "--gpu", "auto", "--context-length", "4096"], timeout=600)
        if load_result.returncode != 0 and model_key not in _models_at(base_url):
            detail = (load_result.stderr or load_result.stdout or server_result.stderr or "").strip()
            raise RuntimeError(detail[-1200:] or "LM Studio could not load this model.")

    deadline = time.monotonic() + 180
    models: List[str] = []
    while time.monotonic() < deadline:
        models = _models_at(base_url)
        if models:
            return base_url, models
        time.sleep(0.75)
    raise RuntimeError("LM Studio did not make the model available. Check that the model fits in your computer's memory.")


def ensure_llama_server(progress=None) -> Path:
    use_cuda = has_nvidia_gpu()
    target_dll = "ggml-cuda.dll" if use_cuda else "ggml-vulkan.dll"
    if LLAMA_EXE.is_file() and (LLAMA_DIR / target_dll).is_file():
        return LLAMA_EXE
    from core.llama_tts import (
        LLAMA_TTS_DIR,
        LLAMA_TTS_SERVER_EXE,
        resolve_llama_cpp_cuda_asset_urls,
        resolve_llama_cpp_win_asset_url,
    )

    if LLAMA_TTS_SERVER_EXE.is_file() and (LLAMA_TTS_DIR / target_dll).is_file():
        return LLAMA_TTS_SERVER_EXE
    if platform.machine().lower() not in ("amd64", "x86_64", "x64"):
        raise RuntimeError("Automatic llama.cpp setup currently supports Windows x64 only.")
    if progress:
        progress(tr("ui_local_models_gguf_downloading", "Завантажую офіційний GPU-модуль llama.cpp…"))
    LLAMA_DIR.mkdir(parents=True, exist_ok=True)
    from core.tts_manager import tts_manager

    def _safe_extract(archive_path: Path) -> None:
        with zipfile.ZipFile(archive_path, "r") as zf:
            root = LLAMA_DIR.resolve()
            for member in zf.infolist():
                destination = (LLAMA_DIR / member.filename).resolve()
                if destination != root and root not in destination.parents:
                    raise RuntimeError("The downloaded llama.cpp archive contains an invalid file path.")
            zf.extractall(LLAMA_DIR)
        archive_path.unlink(missing_ok=True)

    if use_cuda:
        cuda_url, cudart_url = resolve_llama_cpp_cuda_asset_urls()
        archive = LLAMA_DIR / "llama-bin-win-cuda-x64.zip"
        tts_manager._download_url_with_progress(
            cuda_url, archive, "local_llama_cpp", 0, 60,
            "llama.cpp NVIDIA CUDA server",
        )
        if progress:
            progress(tr("ui_local_models_gguf_preparing", "Preparing llama.cpp…"))
        _safe_extract(archive)
        if cudart_url and not (LLAMA_DIR / "cublas64_12.dll").is_file():
            cudart_archive = LLAMA_DIR / "cudart-llama-bin-win-cuda-x64.zip"
            tts_manager._download_url_with_progress(
                cudart_url, cudart_archive, "local_llama_cpp", 60, 40,
                "NVIDIA CUDA Runtime DLLs",
            )
            _safe_extract(cudart_archive)
    else:
        archive = LLAMA_DIR / "llama-bin-win-vulkan-x64.zip"
        url = resolve_llama_cpp_win_asset_url(prefer_vulkan=True) or LLAMA_ARCHIVE_URL
        tts_manager._download_url_with_progress(
            url, archive, "local_llama_cpp", 0, 100,
            "llama.cpp Vulkan GPU server",
        )
        if progress:
            progress(tr("ui_local_models_gguf_preparing", "Preparing llama.cpp…"))
        _safe_extract(archive)

    if not LLAMA_EXE.is_file():
        raise RuntimeError("The llama.cpp package was unpacked, but llama-server.exe was not found.")
    return LLAMA_EXE


def _close_process_entry(entry: Dict[str, Any]) -> None:
    process = entry.get("process")
    if process is not None and process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=4)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
    log_handle = entry.get("log")
    if log_handle is not None:
        try:
            log_handle.close()
        except Exception:
            pass


def _launch_llama_attempt(
    exe: Path,
    model: Path,
    chosen_port: int,
    use_gpu: bool,
    log_path: Path,
) -> Tuple[str, List[str]]:
    base_url = f"http://127.0.0.1:{chosen_port}/v1"
    if _models_at(base_url):
        raise RuntimeError(f"Port {chosen_port} is already being used by another local AI server.")
    ngl = "99" if use_gpu else "0"
    with _PROCESS_LOCK:
        log_handle = open(log_path, "ab")
        process = subprocess.Popen(
            [
                str(exe),
                "-m", str(model),
                "--host", "127.0.0.1",
                "--port", str(chosen_port),
                "-c", "4096",
                "-ngl", ngl,
            ],
            cwd=str(exe.parent),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            **_hidden_process_options(),
        )
        assign_process_to_kill_on_close_job(process)
        _LLAMA_PROCESSES[str(model)] = {"process": process, "url": base_url, "log": log_handle}

    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        with _PROCESS_LOCK:
            proc_now = _LLAMA_PROCESSES.get(str(model), {}).get("process")
        if proc_now is None or proc_now.poll() is not None:
            break
        models = _models_at(base_url)
        if models:
            return base_url, models
        time.sleep(0.75)

    with _PROCESS_LOCK:
        failed_entry = _LLAMA_PROCESSES.pop(str(model), None)
    if failed_entry:
        _close_process_entry(failed_entry)
    log_tail = ""
    try:
        if log_path.is_file():
            log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-1200:]
    except Exception:
        pass
    raise RuntimeError(log_tail.strip() or "llama.cpp did not start. The model may be too large for available memory.")


def start_llama_cpp_server(model_path: str, port: Optional[int] = None, progress=None) -> Tuple[str, List[str]]:
    model = Path(model_path).expanduser().resolve()
    if not model.is_file() or model.suffix.lower() != ".gguf":
        raise RuntimeError("Choose a valid GGUF model file.")
    exe = ensure_llama_server(progress)
    with _PROCESS_LOCK:
        entry = _LLAMA_PROCESSES.get(str(model))
        if entry and entry["process"].poll() is None:
            url = str(entry["url"])
            models = _models_at(url)
            if models:
                return url, models
        if entry:
            _LLAMA_PROCESSES.pop(str(model), None)
            _close_process_entry(entry)

    chosen_port = int(port or 0)
    if not chosen_port:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            chosen_port = int(probe.getsockname()[1])
    if progress:
        progress(tr("ui_local_models_gguf_launching", "Starting the model. First startup can take a little while…"))
    log_path = DATA_DIR / "llama-server.log"

    has_gpu_backend = (exe.parent / "ggml-cuda.dll").is_file() or (exe.parent / "ggml-vulkan.dll").is_file()
    if has_gpu_backend:
        try:
            return _launch_llama_attempt(exe, model, chosen_port, use_gpu=True, log_path=log_path)
        except Exception as exc:
            print(f"[LocalLLM] GPU startup failed, retrying on CPU: {exc}", flush=True)
    return _launch_llama_attempt(exe, model, chosen_port, use_gpu=False, log_path=log_path)


def stop_managed_llama_server() -> None:
    with _PROCESS_LOCK:
        entries = list(_LLAMA_PROCESSES.values())
        _LLAMA_PROCESSES.clear()
        for entry in entries:
            _close_process_entry(entry)


def unload_saved_local_models(items: List[Dict[str, Any]]) -> None:
    """Unload configured Ollama and LM Studio models when SilentVoice exits."""
    seen: set[Tuple[str, str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        provider = str(item.get("provider", "")).strip().lower()
        model = str(item.get("model", "")).strip()
        if not model:
            continue

        if provider == "ollama":
            base_url = str(item.get("base_url") or "http://localhost:11434").strip().rstrip("/")
            try:
                host = urlsplit(base_url).hostname
            except ValueError:
                continue
            # SilentVoice only manages the local Ollama instance. Avoid sending
            # an unload command to a remote service that may be shared.
            if host not in ("localhost", "127.0.0.1", "::1"):
                continue
            key = (provider, base_url, model)
            if key in seen:
                continue
            seen.add(key)
            try:
                response = requests.post(
                    f"{base_url}/api/chat",
                    json={"model": model, "messages": [], "keep_alive": 0, "stream": False},
                    timeout=3,
                )
                response.raise_for_status()
            except Exception as exc:
                try:
                    resp2 = requests.post(
                        f"{base_url}/api/generate",
                        json={"model": model, "prompt": "", "keep_alive": 0, "stream": False},
                        timeout=3,
                    )
                    resp2.raise_for_status()
                except Exception:
                    print(f"[LocalLLM] Could not unload Ollama model {model}: {exc}", flush=True)
        elif provider == "local_api" and str(item.get("managed_backend", "")).strip().lower() == "lmstudio":
            key = ("lmstudio", "", model)
            if key in seen:
                continue
            seen.add(key)
            try:
                result = _run_lms(["unload", model], timeout=10)
                if result.returncode != 0:
                    fallback_res = _run_lms(["unload", "--all"], timeout=10)
                    if fallback_res.returncode != 0:
                        detail = (result.stderr or result.stdout or "").strip()
                        if detail:
                            print(f"[LocalLLM] Could not unload LM Studio model {model}: {detail[-500:]}", flush=True)
            except Exception as exc:
                print(f"[LocalLLM] Could not unload LM Studio model {model}: {exc}", flush=True)


atexit.register(stop_managed_llama_server)
