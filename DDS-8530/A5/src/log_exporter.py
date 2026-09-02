"""Tail logs/ and runtime/logs/ and expose per-file line/level counts for Prometheus."""
import logging
import re
import time
from pathlib import Path

from prometheus_client import Counter, Gauge, start_http_server

PROJECT_DIR = Path(__file__).resolve().parents[1]
WATCH_DIRS = [PROJECT_DIR / "logs", PROJECT_DIR / "runtime/logs"]
POLL_INTERVAL_S = 2
EXPORTER_PORT = 8001

LEVEL_PATTERN = re.compile(r"\b(DEBUG|INFO|WARNING|ERROR|CRITICAL)\b")

LOG_LINES = Counter("log_lines_total", "Log lines observed per file and level", ["file", "level"])
LOG_FILE_SIZE = Gauge("log_file_size_bytes", "Current size of a watched log file", ["file"])

logger = logging.getLogger(__name__)


def _classify(line: str) -> str:
    match = LEVEL_PATTERN.search(line)
    return match.group(1) if match else "unknown"


class LogTailer:
    """Tracks one log file's read offset and reports newly appended lines as metrics."""

    def __init__(self, path: Path):
        self.path = path
        self.name = str(path.relative_to(PROJECT_DIR))
        self._offset = path.stat().st_size

    def poll(self) -> None:
        if not self.path.exists():
            return
        size = self.path.stat().st_size
        LOG_FILE_SIZE.labels(file=self.name).set(size)
        if size < self._offset:
            self._offset = 0  # rotated or truncated since the last poll
        with self.path.open("r", errors="replace") as f:
            f.seek(self._offset)
            for line in f:
                LOG_LINES.labels(file=self.name, level=_classify(line)).inc()
            self._offset = f.tell()


def discover_log_files() -> list[Path]:
    return [path for directory in WATCH_DIRS if directory.is_dir() for path in sorted(directory.glob("*.log"))]


def run() -> None:
    logging.basicConfig(level=logging.INFO)
    start_http_server(EXPORTER_PORT)
    logger.info("log_exporter listening port=%d watching=%s", EXPORTER_PORT, [str(d) for d in WATCH_DIRS])

    tailers: dict[Path, LogTailer] = {}
    while True:
        for path in discover_log_files():
            if path not in tailers:
                tailers[path] = LogTailer(path)
                logger.info("log_exporter tracking file=%s", tailers[path].name)
        for tailer in tailers.values():
            tailer.poll()
        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    run()
