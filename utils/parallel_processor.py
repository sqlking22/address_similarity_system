#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2025/12/5 15:15
# @Author  : hejun
"""
并行处理模块
优化百万级数据处理性能
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Callable, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import time
from functools import partial
import psutil


class ParallelProcessor:
    """并行处理器"""

    def __init__(self, n_jobs: Optional[int] = None,
                 max_memory_gb: Optional[float] = None):
        """
        初始化并行处理器

        Args:
            n_jobs: 并行任务数，None表示自动检测
            max_memory_gb: 最大内存使用（GB），None表示无限制
        """
        if n_jobs is None:
            # 自动检测：使用CPU核心数-2，至少为1
            self.n_jobs = max(1, mp.cpu_count() - 2)
        else:
            self.n_jobs = max(1, n_jobs)

        self.max_memory_gb = max_memory_gb
        self.processes = []

    def batch_process(self,
                      data: List[Any],
                      process_func: Callable,
                      batch_size: int = 1000,
                      desc: str = "Processing") -> List[Any]:
        """
        批量并行处理数据

        Args:
            data: 数据列表
            process_func: 处理函数
            batch_size: 批处理大小
            desc: 进度描述

        Returns:
            处理结果列表
        """
        total_items = len(data)
        results = []

        print(f"{desc}: 共 {total_items:,} 条数据，批处理大小: {batch_size}")

        # 分批处理
        for batch_start in range(0, total_items, batch_size):
            batch_end = min(batch_start + batch_size, total_items)
            batch = data[batch_start:batch_end]

            print(f"处理批次 {batch_start // batch_size + 1}/{(total_items + batch_size - 1) // batch_size} "
                  f"({batch_start:,} - {batch_end:,})")

            # 检查内存使用
            if self.max_memory_gb:
                self._check_memory_usage()

            # 并行处理当前批次
            batch_results = self._parallel_process_batch(batch, process_func)
            results.extend(batch_results)

        return results

    def _parallel_process_batch(self, batch: List[Any],
                                process_func: Callable) -> List[Any]:
        """
        并行处理单个批次
        """
        if self.n_jobs == 1 or len(batch) < 100:
            # 小批量或单进程模式
            return [process_func(item) for item in batch]

        # 多进程模式
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # 提交任务
            future_to_item = {
                executor.submit(process_func, item): item
                for item in batch
            }

            # 收集结果
            batch_results = []
            for future in as_completed(future_to_item):
                try:
                    result = future.result()
                    batch_results.append(result)
                except Exception as e:
                    item = future_to_item[future]
                    print(f"处理失败: {item}, 错误: {e}")
                    batch_results.append(None)

            return batch_results

    def map_reduce(self,
                   data: List[Any],
                   map_func: Callable,
                   reduce_func: Callable,
                   chunk_size: int = 1000) -> Any:
        """
        Map-Reduce模式处理数据

        Args:
            data: 数据列表
            map_func: 映射函数
            reduce_func: 归约函数
            chunk_size: 块大小

        Returns:
            归约结果
        """
        total_items = len(data)

        print(f"Map-Reduce处理: 共 {total_items:,} 条数据")

        # 第一步：Map（并行）
        mapped_results = self.batch_process(
            data, map_func, batch_size=chunk_size, desc="Map阶段"
        )

        # 第二步：Reduce
        print("Reduce阶段...")
        result = reduce_func(mapped_results)

        return result

    def process_dataframe(self,
                          df: pd.DataFrame,
                          process_func: Callable,
                          column: str,
                          result_column: str = None,
                          batch_size: int = 1000) -> pd.DataFrame:
        """
        并行处理DataFrame列

        Args:
            df: 输入DataFrame
            process_func: 处理函数
            column: 要处理的列名
            result_column: 结果列名，None则覆盖原列
            batch_size: 批处理大小

        Returns:
            处理后的DataFrame
        """
        if result_column is None:
            result_column = column

        # 提取数据
        data = df[column].tolist()

        # 并行处理
        results = self.batch_process(
            data, process_func, batch_size, f"处理列 '{column}'"
        )

        # 添加结果到DataFrame
        df[result_column] = results

        return df

    def _check_memory_usage(self):
        """检查内存使用"""
        if self.max_memory_gb:
            memory_info = psutil.virtual_memory()
            used_gb = memory_info.used / (1024 ** 3)

            if used_gb > self.max_memory_gb * 0.9:  # 达到90%阈值
                print(f"警告: 内存使用过高 ({used_gb:.1f}GB)，暂停处理...")
                time.sleep(5)  # 暂停5秒

    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        memory = psutil.virtual_memory()
        cpu_count = mp.cpu_count()

        return {
            'cpu_count': cpu_count,
            'cpu_usage': psutil.cpu_percent(),
            'memory_total_gb': memory.total / (1024 ** 3),
            'memory_available_gb': memory.available / (1024 ** 3),
            'memory_used_percent': memory.percent,
            'n_jobs': self.n_jobs,
            'max_memory_gb': self.max_memory_gb
        }


class MemoryOptimizedProcessor:
    """内存优化处理器"""

    def __init__(self, max_memory_gb: float = 32):
        """
        初始化内存优化处理器

        Args:
            max_memory_gb: 最大内存使用（GB）
        """
        self.max_memory_gb = max_memory_gb

    def process_large_file(self,
                           filepath: str,
                           process_func: Callable,
                           chunksize: int = 10000,
                           **kwargs) -> pd.DataFrame:
        """
        处理大文件（分块读取和处理）

        Args:
            filepath: 文件路径
            process_func: 处理函数
            chunksize: 块大小
            **kwargs: 传递给read_csv的参数

        Returns:
            处理后的DataFrame
        """
        results = []
        chunk_iter = 0

        # 分块读取和处理
        for chunk in pd.read_csv(filepath, chunksize=chunksize, **kwargs):
            chunk_iter += 1
            print(f"处理块 {chunk_iter} (共 {len(chunk)} 行)")

            # 处理当前块
            processed_chunk = process_func(chunk)
            results.append(processed_chunk)

            # 检查内存
            self._check_and_cleanup(results)

        # 合并所有块
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()

    def _check_and_cleanup(self, data_chunks: List[pd.DataFrame]):
        """检查内存并清理"""
        # 估算当前内存使用
        estimated_memory_mb = sum(chunk.memory_usage(deep=True).sum() / 1024 / 1024
                                  for chunk in data_chunks)

        if estimated_memory_mb > self.max_memory_gb * 1024 * 0.8:  # 达到80%阈值
            print(f"内存警告: 当前使用约 {estimated_memory_mb:.1f}MB，达到阈值")

            # 可以在这里实现保存中间结果到磁盘
            # 或者删除早期的不需要的块

    def streaming_process(self,
                          data_generator,
                          process_func: Callable,
                          batch_size: int = 1000) -> List[Any]:
        """
        流式处理数据

        Args:
            data_generator: 数据生成器
            process_func: 处理函数
            batch_size: 批处理大小

        Returns:
            处理结果列表
        """
        results = []
        current_batch = []

        processor = ParallelProcessor()

        for item in data_generator:
            current_batch.append(item)

            if len(current_batch) >= batch_size:
                # 处理当前批次
                batch_results = processor.batch_process(
                    current_batch, process_func, batch_size=len(current_batch)
                )
                results.extend(batch_results)
                current_batch = []

                # 检查内存
                self._check_memory_usage()

        # 处理剩余数据
        if current_batch:
            batch_results = processor.batch_process(
                current_batch, process_func, batch_size=len(current_batch)
            )
            results.extend(batch_results)

        return results

    def _check_memory_usage(self):
        """检查内存使用"""
        memory = psutil.virtual_memory()
        used_gb = memory.used / (1024 ** 3)

        if used_gb > self.max_memory_gb * 0.9:
            print(f"内存使用过高: {used_gb:.1f}GB / {self.max_memory_gb:.1f}GB")


class SimilarityMatrixBuilder:
    """相似度矩阵构建器（内存优化）"""

    def __init__(self, max_memory_gb: float = 32):
        self.max_memory_gb = max_memory_gb

    def build_sparse_similarity_matrix(self,
                                       n_items: int,
                                       similarity_pairs: List[Tuple[int, int, float]],
                                       threshold: float = 0.0) -> csr_matrix:
        """
        构建稀疏相似度矩阵

        Args:
            n_items: 项目数量
            similarity_pairs: 相似度对列表 (i, j, similarity)
            threshold: 相似度阈值

        Returns:
            稀疏相似度矩阵
        """
        from scipy.sparse import csr_matrix

        # 过滤低于阈值的对
        filtered_pairs = [(i, j, sim) for i, j, sim in similarity_pairs
                          if sim >= threshold]

        if not filtered_pairs:
            return csr_matrix((n_items, n_items))

        # 提取行列和数据
        rows, cols, data = zip(*filtered_pairs)

        # 创建对称矩阵（添加对称元素）
        rows_ext = list(rows) + list(cols)
        cols_ext = list(cols) + list(rows)
        data_ext = list(data) + list(data)

        return csr_matrix((data_ext, (rows_ext, cols_ext)),
                          shape=(n_items, n_items))

    def build_block_diagonal_matrix(self,
                                    blocks: List[np.ndarray]) -> csr_matrix:
        """
        构建块对角矩阵（用于分布式计算）

        Args:
            blocks: 块矩阵列表

        Returns:
            块对角矩阵
        """
        from scipy.sparse import block_diag
        return block_diag(blocks)

    def save_matrix_to_disk(self,
                            matrix: csr_matrix,
                            filepath: str):
        """保存矩阵到磁盘"""
        import pickle

        with open(filepath, 'wb') as f:
            pickle.dump(matrix, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_matrix_from_disk(self, filepath: str) -> csr_matrix:
        """从磁盘加载矩阵"""
        import pickle

        with open(filepath, 'rb') as f:
            return pickle.load(f)