"""
IndicSent-M: Unified Application Launcher
==========================================
Starts both the FastAPI Backend and the React Frontend simultaneously.
When you press Ctrl+C (or close the terminal window), both servers are cleanly terminated.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"


def kill_process_tree(pid: int):
    """Cleanly and forcefully terminate process and its entire child process tree on Windows/Linux."""
    if not pid:
        return
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False
            )
        else:
            import signal
            os.killpg(os.getpgid(pid), signal.SIGTERM)
    except Exception:
        pass


def main():
    print("=" * 68)
    print("                 IndicSent-M: Unified Launcher")
    print("=" * 68)
    print("  [1/2] Launching FastAPI Backend on http://127.0.0.1:8000 ...")

    # 1. Start Backend using the active Python executable
    backend_cmd = [
        sys.executable, "-m", "uvicorn", "app.main:app",
        "--host", "127.0.0.1", "--port", "8000"
    ]
    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=str(BACKEND_DIR)
    )

    # 2. Start Frontend
    print("  [2/2] Launching React Frontend on http://localhost:3000 ...")
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    frontend_cmd = [npm_cmd, "run", "dev"]
    frontend_proc = subprocess.Popen(
        frontend_cmd,
        cwd=str(PROJECT_ROOT)
    )

    print("-" * 68)
    print("  -> Backend API:     http://127.0.0.1:8000 (Swagger: /docs)")
    print("  -> Frontend UI:     http://localhost:3000")
    print("-" * 68)
    print("  IndicSent-M is running!")
    print("  >>> Press Ctrl+C at any time to STOP the entire project. <<<\n")

    try:
        while True:
            time.sleep(0.5)
            # Check if any process terminated unexpectedly
            if backend_proc.poll() is not None:
                print("\n[Notice] Backend process exited.")
                break
            if frontend_proc.poll() is not None:
                print("\n[Notice] Frontend process exited.")
                break
    except KeyboardInterrupt:
        print("\n\nStopping IndicSent-M project...")
    finally:
        print(f"Terminating Backend server (PID {backend_proc.pid})...")
        kill_process_tree(backend_proc.pid)
        print(f"Terminating Frontend server (PID {frontend_proc.pid})...")
        kill_process_tree(frontend_proc.pid)
        print("=" * 68)
        print("  All services stopped cleanly. Ports 8000 & 3000 released.")
        print("=" * 68)


if __name__ == "__main__":
    main()
