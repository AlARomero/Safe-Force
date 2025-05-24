import logging
from datetime import datetime
import sys


class ExecutionLogger:
    def __init__(self, log_file_path: str):
        self.start_time = datetime.now()
        self.last_log_time = self.start_time
        self.log_file_path = log_file_path
        if self.log_file_path:
            with open(self.log_file_path, 'w', encoding='utf-8') as f:
                f.write(f"[LOG STARTED] {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

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
            if self.log_file_path:
                with open(self.log_file_path, 'a', encoding='utf-8') as f:
                    f.write(log_msg + '\n')
            self.last_log_time = current_time

def get_basic_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.hasHandlers():
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    return logger