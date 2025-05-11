from datetime import datetime
import sys


class ExecutionLogger:
    def __init__(self):
        self.start_time = datetime.now()
        self.last_log_time = self.start_time

    def log(self, message: str, level: str = "INFO"):
        current_time = datetime.now()
        elapsed = (current_time - self.start_time).total_seconds()
        header = {
            "INFO": "[ℹ️]",
            "WARNING": "[⚠️]",
            "SUCCESS": "[✔️]",
            "ERROR": "[❌]",
            "DEBUG": "[🐛]"
        }.get(level, "[❔]")
        log_msg = f"{header} [T+{elapsed:.1f}s] {message}"
        print(log_msg)
        sys.stdout.flush()
        self.last_log_time = current_time