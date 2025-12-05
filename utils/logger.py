#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2025/12/5 15:33
# @Author  : hejun
"""
日志记录工具
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class Logger:
    """日志记录器"""

    def __init__(self, name: str = 'address_similarity',
                 log_dir: Optional[str] = None,
                 level: int = logging.INFO,
                 console: bool = True):
        """
        初始化日志记录器

        Args:
            name: 日志记录器名称
            log_dir: 日志目录，None表示不保存到文件
            level: 日志级别
            console: 是否输出到控制台
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()  # 清除现有处理器

        # 设置格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 控制台处理器
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # 文件处理器
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_dir / f"{name}_{timestamp}.log"

            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            self.log_file = log_file
        else:
            self.log_file = None

    def get_logger(self) -> logging.Logger:
        """获取日志记录器实例"""
        return self.logger

    def info(self, message: str):
        """记录信息"""
        self.logger.info(message)

    def debug(self, message: str):
        """记录调试信息"""
        self.logger.debug(message)

    def warning(self, message: str):
        """记录警告"""
        self.logger.warning(message)

    def error(self, message: str):
        """记录错误"""
        self.logger.error(message)

    def critical(self, message: str):
        """记录严重错误"""
        self.logger.critical(message)

    def log_progress(self, current: int, total: int,
                     prefix: str = "进度", step: int = 1000):
        """记录进度"""
        if current % step == 0 or current == total:
            percentage = (current / total) * 100
            self.info(f"{prefix}: {current}/{total} ({percentage:.1f}%)")

    def get_log_file(self) -> Optional[Path]:
        """获取日志文件路径"""
        return self.log_file


def setup_logging(name: str = 'address_similarity',
                  log_dir: Optional[str] = None,
                  level: int = logging.INFO) -> Logger:
    """
    快速设置日志记录

    Args:
        name: 日志记录器名称
        log_dir: 日志目录
        level: 日志级别

    Returns:
        日志记录器实例
    """
    return Logger(name, log_dir, level)