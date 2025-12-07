# !/usr/bin/env python3
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


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""

    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',  # 青色
        'INFO': '\033[32m',  # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',  # 红色
        'CRITICAL': '\033[35m',  # 紫色
        'RESET': '\033[0m'  # 重置
    }

    def __init__(self, fmt: str, datefmt: str = None):
        super().__init__(fmt, datefmt)
        self.fmt = fmt

    def format(self, record):
        # 添加行号信息
        log_fmt = self.fmt
        if hasattr(record, 'lineno'):
            log_fmt = log_fmt.replace('%(message)s', '[%(lineno)d] %(message)s')

        # 为警告和错误级别添加颜色
        if record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            reset = self.COLORS['RESET']
            record.levelname = f"{color}{record.levelname}{reset}"

        formatter = logging.Formatter(log_fmt, self.datefmt)
        return formatter.format(record)


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

        # 设置格式（包含行号）
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 彩色控制台格式化器（包含行号）
        colored_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 控制台处理器
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(colored_formatter)  # 使用彩色格式化器
            self.logger.addHandler(console_handler)

        # 文件处理器
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_dir / f"{name}_{timestamp}.log"

            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)  # 文件使用普通格式化器
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
