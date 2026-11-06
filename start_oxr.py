from __future__ import annotations

import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config.yaml"
LOG_DIR = ROOT / "logs"
LAYOUT_DIR = ROOT / "services" / "layout_analyze"
MODEL_DIR = ROOT / "models" / "OXR-1.0" / "layout_analysis"
DEFAULT_API_PORT = 8000
DEFAULT_LAYOUT_PORT = 8081
PROBE_TIMEOUT = 3

# Bypass machine proxy settings for localhost and the configured intranet model.
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
started: list[tuple[str, subprocess.Popen, object]] = []


def config_value(section: str, key: str, default: str) -> str:
    """Read a scalar from the simple two-space-indented config.yaml sections."""
    if not CONFIG.is_file():
        raise RuntimeError(f"找不到配置文件：{CONFIG}")
    lines = CONFIG.read_text(encoding="utf-8-sig").splitlines()
    in_section = False
    pattern = re.compile(r"^  " + re.escape(key) + r"\s*:\s*(.*?)\s*$")
    for line in lines:
        if line and not line[0].isspace():
            in_section = line.rstrip().rstrip(":") == section
            continue
        if in_section:
            match = pattern.match(line)
            if match:
                value = match.group(1).split(" #", 1)[0].strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                return value
    return default


def python_in_venv(folder: Path) -> Path:
    candidates = [folder / "Scripts" / "python.exe", folder / "bin" / "python"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        f"找不到虚拟环境 Python：{folder}\n"
        "请先完成 OXR 和 services/layout_analyze 的依赖安装。"
    )


def http_json(url: str, timeout: float = PROBE_TIMEOUT):
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with HTTP.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def http_ok(url: str, timeout: float = PROBE_TIMEOUT) -> bool:
    try:
        with HTTP.open(url, timeout=timeout) as response:
            return 200 <= response.status < 300
    except (OSError, urllib.error.URLError, TimeoutError):
        return False


def port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def safe_url(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    host = parts.hostname or ""
    if parts.port:
        host += f":{parts.port}"
    return urllib.parse.urlunsplit((parts.scheme, host, parts.path, "", ""))


def tail(path: Path, count: int = 24) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:])
    except OSError:
        return "(日志暂不可读)"


def launch(name: str, command: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> subprocess.Popen:
    log_file = log_path.open("a", encoding="utf-8", errors="replace")
    log_file.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} 启动 {name} ---\n")
    log_file.flush()
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    try:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )
    except Exception:
        log_file.close()
        raise
    started.append((name, process, log_file))
    print(f"已启动 {name}（PID {process.pid}）；日志：{log_path}")
    return process


def wait_ready(name: str, url: str, process: subprocess.Popen, timeout: int, log_path: Path, check_json: bool = False) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"{name} 启动失败（退出码 {process.returncode}）。\n"
                f"日志末尾：\n{tail(log_path)}"
            )
        try:
            if check_json:
                data = http_json(url, timeout=2)
                if data.get("status") == "healthy":
                    print(f"{name} 已就绪。")
                    return
            elif http_ok(url, timeout=2):
                print(f"{name} 已就绪。")
                return
        except (OSError, ValueError, urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise RuntimeError(
        f"等待 {name} 就绪超时（{timeout} 秒）：{url}\n"
        f"日志末尾：\n{tail(log_path)}"
    )


def verify_intranet_model(base_url: str, expected_model: str) -> None:
    models_url = base_url.rstrip("/") + "/models"
    try:
        data = http_json(models_url, timeout=8)
    except Exception as exc:
        raise RuntimeError(
            f"无法连接内网 OXR 模型服务：{safe_url(models_url)}\n"
            "请确认已连入内网、服务正在运行，并检查 config.yaml 中的 oxr_model.url。"
        ) from exc
    available = [item.get("id", "") for item in data.get("data", []) if isinstance(item, dict)]
    if expected_model not in available:
        raise RuntimeError(
            f"内网模型服务可连接，但未找到配置的模型 {expected_model!r}。"
            f"当前可用模型：{', '.join(available) or '(无)'}"
        )
    print(f"内网 OXR 模型可用：{expected_model}（{safe_url(base_url)}）")


def start_layout(layout_port: int, layout_python: Path) -> None:
    base = f"http://127.0.0.1:{layout_port}"
    health_url = base + "/health"
    log_path = LOG_DIR / "layout_analyze.log"
    if http_ok(health_url):
        try:
            if http_json(health_url).get("status") == "healthy":
                print(f"本机版面分析服务已在运行：{base}")
                return
        except Exception:
            pass
    if port_is_open("127.0.0.1", layout_port):
        raise RuntimeError(f"版面分析端口 {layout_port} 已被其他服务占用，且 /health 未通过。")

    checkpoint = MODEL_DIR / "last.pth"
    metadata = MODEL_DIR / "metadata.json"
    if not checkpoint.is_file() or not metadata.is_file():
        raise RuntimeError(
            "缺少本机版面分析模型文件：\n"
            f"{checkpoint}\n{metadata}\n"
            "请先下载 OXR-1.0 的 layout_analysis 模型文件。"
        )
    env = os.environ.copy()
    env.update({
        "PORT": str(layout_port),
        "MODEL_PATH": str(checkpoint),
        "MODEL_METADATA_PATH": str(metadata),
        "DFINE_DEVICE": "cpu",
    })
    process = launch(
        "本机 CPU 版面分析",
        [str(layout_python), "-m", "dfine_la"],
        LAYOUT_DIR,
        env,
        log_path,
    )
    wait_ready("本机 CPU 版面分析", health_url, process, 180, log_path, check_json=True)


def start_api(api_port: int, api_python: Path) -> str:
    base = f"http://127.0.0.1:{api_port}"
    page_url = base + "/workbench"
    log_path = LOG_DIR / "oxr_api.log"
    if http_ok(page_url):
        print(f"OXR API 与识别前端已在运行：{page_url}")
        return page_url
    if port_is_open("127.0.0.1", api_port):
        raise RuntimeError(f"OXR API 端口 {api_port} 已被其他服务占用，且工作台页面未通过。")
    process = launch(
        "OXR API 与识别前端",
        [str(api_python), "-m", "oxr.cli.main", "serve", "--config", str(CONFIG)],
        ROOT,
        os.environ.copy(),
        log_path,
    )
    wait_ready("OXR API 与识别前端", page_url, process, 90, log_path)
    return page_url


def stop_started() -> None:
    for name, process, log_file in reversed(started):
        if process.poll() is None:
            print(f"正在停止本次启动的 {name}（PID {process.pid}）...")
            process.terminate()
            try:
                process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        log_file.close()


def main() -> int:
    if not CONFIG.is_file():
        print(f"缺少配置文件：{CONFIG}", file=sys.stderr)
        return 1

    try:
        api_port = int(config_value("server", "port", str(DEFAULT_API_PORT)))
        layout_url = config_value("layout_analyze_model", "url", f"http://127.0.0.1:{DEFAULT_LAYOUT_PORT}")
        layout_port = urllib.parse.urlsplit(layout_url).port or DEFAULT_LAYOUT_PORT
        model_url = config_value("oxr_model", "url", "http://192.168.4.80:8093/v1")
        model_name = config_value("oxr_model", "model_name", "OXR-1.0")
        api_python = python_in_venv(ROOT / ".venv")
        layout_python = python_in_venv(LAYOUT_DIR / ".venv")
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        print("OXR 本地识别工作台启动器")
        verify_intranet_model(model_url, model_name)
        start_layout(layout_port, layout_python)
        page_url = start_api(api_port, api_python)
        print(f"\n识别前端：{page_url}")
        print("按 Ctrl+C 停止本次由启动器启动的服务；原本已运行的服务会保留。")
        try:
            webbrowser.open(page_url)
        except Exception as exc:
            print(f"自动打开浏览器失败，请手动访问：{page_url}（{exc}）")

        while True:
            for name, process, _ in started:
                if process.poll() is not None:
                    raise RuntimeError(f"{name} 已退出，退出码 {process.returncode}。")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n收到停止指令。")
    except Exception as exc:
        print(f"\n启动失败：{exc}", file=sys.stderr)
        return 1
    finally:
        stop_started()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
