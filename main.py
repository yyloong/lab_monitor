from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

from lab_monitor.config import load_config
from lab_monitor.notifier.feishu import NOTIFY
from lab_monitor.scheduler import MonitorScheduler


class _ColorFormatter(logging.Formatter):
    """为终端日志添加 ANSI 颜色：NOTIFY 青色，WARNING 黄色，ERROR/CRITICAL 红色，DEBUG 灰色。"""

    _RESET = "\033[0m"
    _COLORS = {
        logging.DEBUG:    "\033[90m",    # 灰色
        logging.INFO:     "",            # 默认
        NOTIFY:           "\033[1;36m",  # 粗体青色（消息推送成功）
        logging.WARNING:  "\033[33m",    # 黄色
        logging.ERROR:    "\033[31m",    # 红色
        logging.CRITICAL: "\033[1;31m",  # 粗体红色
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self._COLORS.get(record.levelno, "")
        msg = super().format(record)
        return f"{color}{msg}{self._RESET}" if color else msg


def _setup_logging(level: int) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_ColorFormatter(fmt=fmt, datefmt=datefmt))

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)


logger = logging.getLogger("lab_monitor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab Monitor — 实验进展自动监控系统")
    parser.add_argument(
        "-c",
        "--config",
        default="config.yaml",
        help="配置文件路径（默认: config.yaml）",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别（默认: INFO）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging(getattr(logging, args.log_level))

    config_path = Path(args.config)
    logger.info("Loading config from: %s", config_path.resolve())
    config = load_config(config_path)

    logger.info(
        "Experiments to monitor: %s",
        [exp.name for exp in config.experiments],
    )

    scheduler = MonitorScheduler(config)

    # 优雅退出
    stop_flag = [False]

    def _handle_signal(signum, frame):  # noqa: ARG001
        logger.info("Received signal %d, shutting down...", signum)
        stop_flag[0] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    scheduler.start()
    logger.info("Lab Monitor running. Press Ctrl+C to stop.")

    try:
        while not stop_flag[0]:
            time.sleep(1)
    finally:
        scheduler.stop()
        logger.info("Lab Monitor stopped.")


if __name__ == "__main__":
    main()
