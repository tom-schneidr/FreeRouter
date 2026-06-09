from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import messagebox

import pystray
from PIL import Image, ImageTk

from app.desktop_icon import build_icon_image
from app.ui.brand import LOGO_PATH

APP_NAME = "FreeRouter"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
APP_USER_MODEL_ID = "FreeRouter.LocalGateway.Console"


class FreeRouterTrayApp:
    def __init__(self, host: str, port: int, reload: bool) -> None:
        self.host = host
        self.port = port
        self.reload = reload
        self.project_root = Path(__file__).resolve().parent.parent
        self.process: subprocess.Popen[str] | None = None
        self.log_queue: queue.Queue[str | None] = queue.Queue()
        self.tray_icon: pystray.Icon | None = None
        self.stopping = False

        self._set_windows_app_id()
        self.root = tk.Tk()
        self.root.title(f"{APP_NAME} Console")
        self.root.geometry("960x560")
        self.root.minsize(720, 420)
        self.root.configure(bg="#0a0e1a")
        self.root.protocol("WM_DELETE_WINDOW", self.hide_to_tray)

        icon_image = self._build_icon_image()
        self.window_icon = ImageTk.PhotoImage(icon_image)
        self.root.iconphoto(True, self.window_icon)
        self.header_logo = ImageTk.PhotoImage(Image.open(LOGO_PATH))

        self.status_var = tk.StringVar(value="Starting FreeRouter...")
        self._build_window()

        self._start_server()
        self._start_tray()
        self.root.after(100, self._drain_logs)
        self.root.after(1000, self._check_process)

    def run(self) -> None:
        self.root.mainloop()

    def _build_window(self) -> None:
        header = tk.Frame(self.root, bg="#111827", padx=16, pady=12)
        header.pack(fill=tk.X)

        title = tk.Label(
            header,
            image=self.header_logo,
            bg="#111827",
        )
        title.pack(side=tk.LEFT)

        subtitle = tk.Label(
            header,
            text=f"  http://{self.host}:{self.port}/v1",
            bg="#111827",
            fg="#93c5fd",
            font=("Consolas", 10),
        )
        subtitle.pack(side=tk.LEFT)

        status = tk.Label(
            header,
            textvariable=self.status_var,
            bg="#111827",
            fg="#94a3b8",
            font=("Segoe UI", 10),
        )
        status.pack(side=tk.RIGHT)

        toolbar = tk.Frame(self.root, bg="#0a0e1a", padx=16, pady=10)
        toolbar.pack(fill=tk.X)

        self._button(toolbar, "Open dashboard", self.open_dashboard).pack(side=tk.LEFT, padx=(0, 8))
        self._button(toolbar, "Open chat", self.open_chat).pack(side=tk.LEFT, padx=(0, 8))
        self._button(toolbar, "Open models", self.open_models).pack(side=tk.LEFT, padx=(0, 8))
        self._button(toolbar, "Open live traffic", self.open_live_traffic).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        self._button(toolbar, "Restart server", self.restart_server).pack(side=tk.LEFT, padx=(0, 8))
        self._button(toolbar, "Hide to tray", self.hide_to_tray).pack(side=tk.LEFT, padx=(0, 8))
        self._button(toolbar, "Stop and exit", self.stop_and_exit).pack(side=tk.LEFT)

        console_frame = tk.Frame(self.root, bg="#0a0e1a", padx=16, pady=(0, 16))
        console_frame.pack(fill=tk.BOTH, expand=True)

        self.console = tk.Text(
            console_frame,
            bg="#020617",
            fg="#dbeafe",
            insertbackground="#dbeafe",
            selectbackground="#1d4ed8",
            relief=tk.FLAT,
            wrap=tk.WORD,
            font=("Consolas", 10),
            padx=12,
            pady=12,
        )
        self.console.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.console.configure(state=tk.DISABLED)

        scroll = tk.Scrollbar(console_frame, command=self.console.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.console.configure(yscrollcommand=scroll.set)

    def _button(self, parent: tk.Widget, text: str, command: Callable[[], None]) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg="#1e293b",
            fg="#e2e8f0",
            activebackground="#334155",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=7,
            cursor="hand2",
        )

    def _start_server(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        args = [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        if self.reload:
            args.append("--reload")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        startupinfo = None
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        self.process = subprocess.Popen(
            args,
            cwd=self.project_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            startupinfo=startupinfo,
        )
        self._append_log(f"Starting FreeRouter at http://{self.host}:{self.port}/v1\n")
        threading.Thread(target=self._read_process_output, daemon=True).start()

    def _stop_server(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            self._append_log("Server did not stop cleanly; forcing shutdown.\n")
            self.process.kill()
            self.process.wait(timeout=3)

    def _read_process_output(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        for line in self.process.stdout:
            self.log_queue.put(line)
        self.log_queue.put(None)

    def _drain_logs(self) -> None:
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if line is None:
                self._append_log("\nFreeRouter server stopped.\n")
                continue
            self._append_log(line)
        if not self.stopping:
            self.root.after(100, self._drain_logs)

    def _append_log(self, text: str) -> None:
        self.console.configure(state=tk.NORMAL)
        self.console.insert(tk.END, text)
        self.console.see(tk.END)
        self.console.configure(state=tk.DISABLED)

    def _check_process(self) -> None:
        if self.process is None:
            return
        code = self.process.poll()
        if code is None:
            self.status_var.set("Running in background")
            self.root.after(1000, self._check_process)
            return
        self.status_var.set(f"Stopped with exit code {code}")

    def _start_tray(self) -> None:
        menu = pystray.Menu(
            pystray.MenuItem(
                "Open console",
                lambda: self.root.after(0, self.show_window),
                default=True,
            ),
            pystray.MenuItem("Open dashboard", lambda: self.root.after(0, self.open_dashboard)),
            pystray.MenuItem("Open chat", lambda: self.root.after(0, self.open_chat)),
            pystray.MenuItem("Open models", lambda: self.root.after(0, self.open_models)),
            pystray.MenuItem("Open live traffic", lambda: self.root.after(0, self.open_live_traffic)),
            pystray.MenuItem("Restart server", lambda: self.root.after(0, self.restart_server)),
            pystray.MenuItem("Stop and exit", lambda: self.root.after(0, self.stop_and_exit)),
        )
        self.tray_icon = pystray.Icon(APP_NAME, self._build_icon_image(), APP_NAME, menu)
        threading.Thread(target=self.tray_icon.run, daemon=True).start()

    def _build_icon_image(self) -> Image.Image:
        return build_icon_image(64)

    def _set_windows_app_id(self) -> None:
        if os.name != "nt":
            return
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
        except Exception:
            pass

    def show_window(self) -> None:
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def hide_to_tray(self) -> None:
        self.root.withdraw()

    def open_dashboard(self) -> None:
        self._open_url("/")

    def open_chat(self) -> None:
        self._open_url("/app#chat")

    def open_models(self) -> None:
        self._open_url("/app#models")

    def open_live_traffic(self) -> None:
        self._open_url("/app#live")

    def _open_url(self, path: str) -> None:
        import webbrowser

        webbrowser.open(f"http://{self.host}:{self.port}{path}")

    def restart_server(self) -> None:
        if self.stopping:
            return
        self.status_var.set("Restarting server...")
        self._append_log("\nRestarting FreeRouter server...\n")
        self._stop_server()
        self._start_server()
        self.root.after(1000, self._check_process)

    def stop_and_exit(self) -> None:
        if self.stopping:
            return
        self.stopping = True
        self._stop_server()
        if self.tray_icon is not None:
            self.tray_icon.stop()
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FreeRouter as a Windows tray app.")
    parser.add_argument("--host", default=os.getenv("GATEWAY_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.getenv("GATEWAY_PORT", DEFAULT_PORT)))
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    try:
        FreeRouterTrayApp(args.host, args.port, args.reload).run()
    except Exception as exc:
        messagebox.showerror(APP_NAME, str(exc))
        raise


if __name__ == "__main__":
    main()
