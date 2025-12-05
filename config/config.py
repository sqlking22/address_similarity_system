#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2025/12/5 15:17
# @Author  : hejun
"""
系统配置文件
"""
import os
from pathlib import Path


class Config:
    """配置类"""

    # 项目路径
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = DATA_DIR / "output"

    # 确保目录存在
    for dir_path in [DATA_DIR, OUTPUT_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # 算法参数
    ALGORITHM_CONFIG = {
        # 相似度权重
        'weights': {
            'text': 0.5,  # 文本相似度权重
            'spatial': 0.3,  # 空间相似度权重
            'admin': 0.2,  # 行政区划相似度权重
        },

        # 文本相似度参数
        'text_similarity': {
            'minhash_permutations': 256,  # MinHash签名长度
            'ngram_size': 3,  # N-Gram大小
            'lsh_threshold': 0.3,  # LSH阈值
        },

        # 空间相似度参数
        'spatial_similarity': {
            'distance_threshold_km': 10,  # 距离阈值（公里）
            'max_search_radius_km': 50,  # 最大搜索半径
        },

        # 聚类参数
        'clustering': {
            'similarity_threshold': 0.7,  # 相似度阈值
            'min_cluster_size': 2,  # 最小聚类大小
            'max_cluster_radius_km': 5,  # 最大聚类半径
        },

        # 性能参数
        'performance': {
            'n_jobs': 28,  # 并行任务数（设置为CPU核心数-4）
            'batch_size': 10000,  # 批处理大小
            'cache_enabled': True,  # 启用缓存
            'use_polars': True,  # 使用Polars加速
        },

        # 地理编码配置
        'geocoding': {
            'use_geocoding_chn': True,  # 使用GeocodingCHN
            'geocoding_chn_path': './GeocodingCHN',  # GeocodingCHN路径
            'fallback_to_api': False,  # API回退
        },

        # 地址标准化配置
        'normalization': {
            'use_cpca': True,  # 使用cpca
            'use_jionlp': True,  # 使用jionlp
            'use_advanced_parser': True,  # 使用高级解析器
        }
    }

    # 输入输出配置
    IO_CONFIG = {
        'input_encoding': 'utf-8',
        'output_encoding': 'utf-8-sig',  # 兼容Excel
        'max_file_size_mb': 1024,  # 最大文件大小
    }

    @classmethod
    def get_output_path(cls, filename):
        """获取输出文件路径"""
        return cls.OUTPUT_DIR / filename

    @classmethod
    def update_config(cls, updates):
        """更新配置"""
        for key, value in updates.items():
            if key in cls.ALGORITHM_CONFIG:
                if isinstance(value, dict):
                    cls.ALGORITHM_CONFIG[key].update(value)
                else:
                    cls.ALGORITHM_CONFIG[key] = value
