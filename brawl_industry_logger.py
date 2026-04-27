import sys
import time as _time
from collections import deque

instance_id = "Main"
for _i, _arg in enumerate(sys.argv):
    if _arg == "--instance" and _i + 1 < len(sys.argv):
        instance_id = sys.argv[_i + 1]
        break

_debug_enabled = False

_log_queue: deque = deque(maxlen=5000)

class _Logger:
    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    @staticmethod
    def _ts() -> str:
        return _time.strftime("%H:%M:%S")

    def _emit(self, level: str, msg):
        line = f"[{self._ts()}] [Instance-{instance_id}] [{level}] {msg}"
        print(line)
        _log_queue.append(line)

    def info(self, msg):    self._emit("INFO ", msg)
    def warning(self, msg): self._emit("WARN ", msg)
    def error(self, msg):   self._emit("ERROR", msg)

    def debug(self, msg):
        if _debug_enabled:
            self._emit("DEBUG", msg)

_loggers: dict = {}

def get_logger(name: str = "brawl_industry") -> _Logger:
    if name not in _loggers:
        _loggers[name] = _Logger(name)
    return _loggers[name]

def set_debug(enabled: bool = False):
    global _debug_enabled
    _debug_enabled = enabled

def pop_logs(max_lines: int = 20) -> list[str]:
    out = []
    while _log_queue and len(out) < max_lines:
        out.append(_log_queue.popleft())
    return out
