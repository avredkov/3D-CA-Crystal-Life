from __future__ import annotations

import sys
import re
from pathlib import Path
import shutil
import logging
from typing import Optional

from PySide6.QtCore import QObject, QProcess
import platform
import subprocess as sp


class SimRunner(QObject):
    def __init__(self, cwd: Optional[Path] = None) -> None:
        super().__init__()
        self.cwd = Path(cwd) if cwd else Path.cwd()
        self.proc: Optional[QProcess] = None
        self._buffer: str = ""
        self.last_percent: Optional[int] = None

    @property
    def is_running(self) -> bool:
        return self.proc is not None and self.proc.state() != QProcess.NotRunning

    def start(self, config_path: Optional[str] = None) -> None:
        if self.is_running:
            raise RuntimeError("Simulation already running")
        p = QProcess(self)
        p.setWorkingDirectory(str(self.cwd))
        p.setProgram(sys.executable)
        args = ["-u", "CA3D.py"]
        if config_path:
            args += ["--config", str(config_path)]
        p.setArguments(args) 
        p.setProcessChannelMode(QProcess.MergedChannels)
        p.readyReadStandardOutput.connect(self._on_read)
        p.readyReadStandardError.connect(self._on_read)
        self._buffer = ""
        self.last_percent = None
        p.start()
        self.proc = p

    def stop(self) -> None:
        if not self.is_running:
            return
        try:
            # Prefer a hard kill to ensure GPU kernels do not block shutdown
            if platform.system().lower().startswith("win"):
                try:
                    pid = int(self.proc.processId())
                    # Kill process tree on Windows
                    sp.call(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=sp.DEVNULL, stderr=sp.DEVNULL)
                except Exception:
                    self.proc.kill()
            else:
                self.proc.terminate()
                self.proc.waitForFinished(2000)
                if self.proc.state() != QProcess.NotRunning:
                    self.proc.kill()
        except Exception:
            pass
        finally:
            self.proc = None

    def _on_read(self) -> None:
        if not self.proc:
            return
        try:
            # Using MergedChannels; read only from standard output to avoid warnings
            data = bytes(self.proc.readAllStandardOutput()).decode(errors="ignore")
            if not data:
                return
            # Keep only the tail to limit memory
            self._buffer = (self._buffer + data)[-4000:]
            # Try to parse tqdm percentage like " 12%|" from the tail
            m = re.search(r"(\d+)%\|", self._buffer)
            if m:
                try:
                    self.last_percent = max(0, min(100, int(m.group(1))))
                except Exception:
                    pass
        except Exception:
            pass




# Module-level logger
logger = logging.getLogger(__name__)


def clear_output_directory_contents(directory: Path) -> None:
    """
    Remove all files and subdirectories within the given directory.

    If the directory does not exist, it will be created.

    Args:
        directory (Path): The target output directory to clear.

    Raises:
        ValueError: If the provided path exists and is not a directory.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        # Create the directory so downstream code can write into it
        dir_path.mkdir(parents=True, exist_ok=True)
        return
    if not dir_path.is_dir():
        raise ValueError(f"Target path is not a directory: {dir_path}")

    for entry in list(dir_path.iterdir()):
        try:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink(missing_ok=True)
        except Exception as exc:
            
            logger.debug("Failed to remove %s: %s", entry, exc)
            # Try a second-chance best-effort for directories
            try:
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
            except Exception:
                pass

