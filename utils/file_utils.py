#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2025/12/5 15:32
# @Author  : hejun
"""
文件处理工具函数
"""
import os
import json
import csv
import pandas as pd
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import gzip
import shutil


class FileUtils:
    """文件处理工具类"""

    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """确保目录存在"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def read_json(filepath: Union[str, Path], encoding: str = 'utf-8') -> Any:
        """读取JSON文件"""
        with open(filepath, 'r', encoding=encoding) as f:
            return json.load(f)

    @staticmethod
    def write_json(data: Any, filepath: Union[str, Path],
                   indent: int = 2, encoding: str = 'utf-8'):
        """写入JSON文件"""
        with open(filepath, 'w', encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)

    @staticmethod
    def read_csv_with_autodetect(filepath: Union[str, Path],
                                 **kwargs) -> pd.DataFrame:
        """自动检测编码读取CSV文件"""
        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'latin1']

        for encoding in encodings:
            try:
                return pd.read_csv(filepath, encoding=encoding, **kwargs)
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue

        raise ValueError(f"无法解码文件: {filepath}")

    @staticmethod
    def save_dataframe(df: pd.DataFrame, filepath: Union[str, Path],
                       index: bool = False, **kwargs):
        """保存DataFrame到文件，自动选择格式"""
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()

        if suffix == '.csv':
            df.to_csv(filepath, index=index, encoding='utf-8-sig', **kwargs)
        elif suffix == '.parquet':
            df.to_parquet(filepath, index=index, **kwargs)
        elif suffix == '.pkl' or suffix == '.pickle':
            df.to_pickle(filepath, **kwargs)
        elif suffix == '.xlsx' or suffix == '.xls':
            df.to_excel(filepath, index=index, **kwargs)
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")

    @staticmethod
    def save_large_dataframe(df: pd.DataFrame, filepath: Union[str, Path],
                             chunksize: int = 100000, **kwargs):
        """分块保存大数据集"""
        filepath = Path(filepath)

        if len(df) <= chunksize:
            FileUtils.save_dataframe(df, filepath, **kwargs)
            return

        # 分块保存
        for i in range(0, len(df), chunksize):
            chunk = df.iloc[i:i + chunksize]
            chunk_file = filepath.parent / f"{filepath.stem}_part{i // chunksize}{filepath.suffix}"
            FileUtils.save_dataframe(chunk, chunk_file, **kwargs)

        print(f"数据已分块保存到: {filepath.parent}/{filepath.stem}_part*{filepath.suffix}")

    @staticmethod
    def compress_file(input_file: Union[str, Path],
                      output_file: Union[str, Path] = None,
                      remove_original: bool = False):
        """压缩文件"""
        input_file = Path(input_file)

        if output_file is None:
            output_file = input_file.with_suffix(input_file.suffix + '.gz')

        with open(input_file, 'rb') as f_in:
            with gzip.open(output_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        if remove_original:
            input_file.unlink()

        print(f"文件已压缩: {output_file}")

    @staticmethod
    def decompress_file(input_file: Union[str, Path],
                        output_file: Union[str, Path] = None,
                        remove_original: bool = False):
        """解压缩文件"""
        input_file = Path(input_file)

        if output_file is None:
            output_file = input_file.with_suffix('')

        with gzip.open(input_file, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        if remove_original:
            input_file.unlink()

        print(f"文件已解压: {output_file}")

    @staticmethod
    def get_file_info(filepath: Union[str, Path]) -> Dict[str, Any]:
        """获取文件信息"""
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")

        stats = path.stat()

        return {
            'filename': path.name,
            'filepath': str(path.absolute()),
            'size_bytes': stats.st_size,
            'size_mb': stats.st_size / (1024 * 1024),
            'modified_time': pd.Timestamp(stats.st_mtime, unit='s'),
            'created_time': pd.Timestamp(stats.st_ctime, unit='s'),
            'is_file': path.is_file(),
            'is_dir': path.is_dir()
        }

    @staticmethod
    def split_file(input_file: Union[str, Path],
                   output_dir: Union[str, Path],
                   lines_per_file: int = 100000,
                   header: bool = True):
        """分割大文件"""
        input_file = Path(input_file)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(input_file, 'r', encoding='utf-8') as f:
            # 读取表头
            if header:
                header_line = f.readline()

            file_count = 0
            line_count = 0
            output_file = None

            for line in f:
                if line_count % lines_per_file == 0:
                    if output_file:
                        output_file.close()

                    file_count += 1
                    output_path = output_dir / f"{input_file.stem}_part{file_count}{input_file.suffix}"
                    output_file = open(output_path, 'w', encoding='utf-8')

                    if header:
                        output_file.write(header_line)

                output_file.write(line)
                line_count += 1

            if output_file:
                output_file.close()

        print(f"文件已分割为 {file_count} 个部分，保存到: {output_dir}")

    @staticmethod
    def merge_files(input_dir: Union[str, Path],
                    output_file: Union[str, Path],
                    pattern: str = "*.csv",
                    header: bool = True):
        """合并文件"""
        input_dir = Path(input_dir)
        output_file = Path(output_file)

        files = sorted(input_dir.glob(pattern))

        if not files:
            print(f"警告: 在 {input_dir} 中没有找到匹配 {pattern} 的文件")
            return

        with open(output_file, 'w', encoding='utf-8') as out_f:
            for i, file_path in enumerate(files):
                with open(file_path, 'r', encoding='utf-8') as in_f:
                    if i == 0 or not header:
                        # 第一个文件或不需要表头时，写入所有内容
                        out_f.write(in_f.read())
                    else:
                        # 跳过表头
                        in_f.readline()
                        out_f.write(in_f.read())

        print(f"文件已合并: {output_file}")