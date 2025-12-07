#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2025/12/5 15:16
# @Author  : hejun
"""
地理编码集成模块
集成GeocodingCHN和其他地理编码服务
"""
import sys
import os
from typing import Dict, Any, Optional, List
import math
import json
from pathlib import Path
import time
from functools import lru_cache
from utils.logger import setup_logging
import subprocess

# 初始化日志记录器
logger = setup_logging('geocoding_integration.py').get_logger()


class GeocodingCHNIntegration:
    """GeocodingCHN集成"""

    def __init__(self, cache_size: int = 10000):
        """
        初始化地理编码器

        Args:
            cache_size: 缓存大小
        """
        self.cache_enabled = True
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'successful': 0,
            'failed': 0,
            'avg_response_time': 0
        }

        # 检查Java环境
        if not self._check_java_environment():
            logger.error("未检测到Java环境，请安装Java并设置JAVA_HOME环境变量")
            self.geocoder = self._create_mock_geocoder()
            return

        # 尝试导入GeocodingCHN
        self.geocoder = self._init_geocoding_chn()

        # 初始化缓存
        self._cache = {}
        self.cache_size = cache_size

    def _check_java_environment(self) -> bool:
        """检查Java环境"""
        try:
            result = subprocess.run(['java', '-version'],
                                    capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _init_geocoding_chn(self):
        """初始化GeocodingCHN"""
        try:
            # 尝试正确的导入方式
            try:
                from GeocodingCHN.Geocoding import Geocoding
                logger.info("成功导入GeocodingCHN")
                return Geocoding(
                    strict=False,  # 非严格模式，适配不完整地址
                    # jvm_path=None
                    jvm_path="E:\\LenovoSoftstore\\jdk-21\\bin\\server\\jvm.dll",
                    data_class_path='core/region.dat')

            except ImportError as e:
                logger.error(f"无法导入GeocodingCHN: {e}")
                return self._create_mock_geocoder()

        except Exception as e:
            logger.error(f"初始化GeocodingCHN失败: {e}")
            return self._create_mock_geocoder()

    def _create_mock_geocoder(self):
        """创建模拟地理编码器"""

        class MockGeocoder:
            def normalizing(self, address):
                """模拟地址标准化"""
                return None

            def geocode(self, address):
                """模拟地理编码"""
                return None

            def reverse_geocode(self, lat, lon):
                """模拟逆地理编码"""
                return None

        return MockGeocoder()

    @lru_cache(maxsize=10000)
    def geocode_cached(self, address: str) -> Optional[Dict[str, Any]]:
        """带缓存的地理编码"""
        return self._geocode_internal(address)

    def _geocode_internal(self, address: str) -> Optional[Dict[str, Any]]:
        """地理编码内部实现"""
        self.stats['total_requests'] += 1
        start_time = time.time()

        try:
            # 使用GeocodingCHN的normalizing方法进行地址标准化
            normalized_result = self.geocoder.normalizing(address)

            if normalized_result:
                parsed = {
                    'latitude': None,  # GeocodingCHN主要做地址标准化，不提供经纬度
                    'longitude': None,
                    'formatted_address': address,
                    'province': getattr(normalized_result, 'province', ''),
                    'city': getattr(normalized_result, 'city', ''),
                    'district': getattr(normalized_result, 'district', ''),
                    'street': getattr(normalized_result, 'street', ''),
                    'street_number': getattr(normalized_result, 'roadNum', ''),
                    'confidence': 1.0,  # 标准化结果置信度设为1
                    'source': 'GeocodingCHN',
                    'timestamp': time.time()
                }

                self.stats['successful'] += 1
                response_time = time.time() - start_time
                self.stats['avg_response_time'] = (self.stats['avg_response_time'] * (
                        self.stats['successful'] - 1) + response_time) / self.stats['successful']

                return parsed

        except Exception as e:
            logger.error(f"地理编码失败: {address}, 错误: {e}")

        self.stats['failed'] += 1
        return None

    def geocode(self, address: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        地理编码：地址 -> 标准化地址

        Args:
            address: 地址字符串
            use_cache: 是否使用缓存

        Returns:
            包含标准化地址信息的字典
        """
        if not address or not isinstance(address, str):
            return None

        cache_key = f"geocode:{address.strip().lower()}"

        if use_cache and self.cache_enabled:
            if cache_key in self._cache:
                self.stats['cache_hits'] += 1
                return self._cache[cache_key]

        result = self.geocode_cached(address)

        if result and use_cache and self.cache_enabled:
            # 管理缓存大小
            if len(self._cache) >= self.cache_size:
                # 移除最旧的10%的缓存项
                items_to_remove = int(self.cache_size * 0.1)
                for _ in range(items_to_remove):
                    if self._cache:
                        self._cache.pop(next(iter(self._cache)))

            self._cache[cache_key] = result

        return result

    def calculate_address_similarity(self, address1: str, address2: str) -> float:
        """
        计算两个地址的相似度

        Args:
            address1: 地址1
            address2: 地址2

        Returns:
            相似度分数 (0-1)
        """
        try:
            similarity = self.geocoder.similarity(address1, address2)
            return float(similarity) if similarity is not None else 0.0
        except Exception as e:
            logger.error(f"计算地址相似度失败: {address1} vs {address2}, 错误: {e}")
            return 0.0

    def segment_address(self, address: str, seg_type: str = 'ik') -> List[str]:
        """
        对地址进行分词

        Args:
            address: 地址字符串
            seg_type: 分词类型 ['ik', 'simple', 'smart', 'word']

        Returns:
            分词结果列表
        """
        try:
            segments = self.geocoder.segment(address, seg_type)
            return segments if segments else []
        except Exception as e:
            logger.error(f"地址分词失败: {address}, 错误: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'cache_size': len(self._cache),
            'cache_hit_rate': self.stats['cache_hits'] / max(self.stats['total_requests'], 1),
            'success_rate': self.stats['successful'] / max(self.stats['total_requests'], 1)
        }

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        self.geocode_cached.cache_clear()
        logger.info("缓存已清空")

    def save_cache_to_file(self, filepath: str):
        """保存缓存到文件"""
        try:
            # 只保存可序列化的数据
            serializable_cache = {}
            for key, value in self._cache.items():
                if isinstance(value, dict):
                    serializable_cache[key] = value

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_cache, f, ensure_ascii=False, indent=2)

            logger.info(f"缓存已保存到: {filepath}")

        except Exception as e:
            logger.error(f"保存缓存失败: {e}")

    def load_cache_from_file(self, filepath: str):
        """从文件加载缓存"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    loaded_cache = json.load(f)

                self._cache.update(loaded_cache)
                logger.info(f"从 {filepath} 加载了 {len(loaded_cache)} 个缓存项")

        except Exception as e:
            logger.error(f"加载缓存失败: {e}")


class MultiSourceGeocoder:
    """多源地理编码器（支持GeocodingCHN和API回退）"""

    def __init__(self, primary_source: str = 'geocoding_chn',
                 api_keys: Dict[str, str] = None):
        """
        初始化多源地理编码器

        Args:
            primary_source: 主要数据源
            api_keys: API密钥字典
        """
        self.primary_source = primary_source
        self.api_keys = api_keys or {}

        # 初始化各数据源
        self.sources = {}

        if primary_source == 'geocoding_chn':
            self.sources['geocoding_chn'] = GeocodingCHNIntegration()

        # 统计信息
        self.stats = {source: {'requests': 0, 'success': 0}
                      for source in self.sources}

    def geocode(self, address: str, sources: List[str] = None) -> Optional[Dict[str, Any]]:
        """
        地理编码，支持多个数据源

        Args:
            address: 地址字符串
            sources: 使用的数据源列表，None表示使用所有可用源

        Returns:
            地理编码结果
        """
        if sources is None:
            sources = list(self.sources.keys())

        # 按优先级尝试各个数据源
        for source in sources:
            if source in self.sources:
                try:
                    self.stats[source]['requests'] += 1
                    result = self.sources[source].geocode(address)

                    if result:
                        self.stats[source]['success'] += 1
                        result['source'] = source
                        return result

                except Exception as e:
                    logger.error(f"数据源 {source} 失败: {e}")

        return None

    def batch_geocode(self, addresses: List[str], n_jobs: int = 8) -> List[Optional[Dict[str, Any]]]:
        """批量地理编码"""
        results = []

        for addr in addresses:
            result = self.geocode(addr)
            results.append(result)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {}
        for source, source_stats in self.stats.items():
            requests = source_stats['requests']
            success = source_stats['success']
            stats[source] = {
                'requests': requests,
                'success': success,
                'success_rate': success / max(requests, 1)
            }
        return stats
