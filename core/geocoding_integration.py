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
from typing import Dict, Any, Optional, List,
import math
import json
from pathlib import Path
import time
from functools import lru_cache


class GeocodingCHNIntegration:
    """GeocodingCHN集成"""

    def __init__(self, data_path: Optional[str] = None, cache_size: int = 10000):
        """
        初始化地理编码器

        Args:
            data_path: GeocodingCHN数据文件路径
            cache_size: 缓存大小
        """
        self.data_path = data_path
        self.cache_enabled = True
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'successful': 0,
            'failed': 0,
            'avg_response_time': 0
        }

        # 尝试导入GeocodingCHN
        self.geocoder = self._init_geocoding_chn()

        # 初始化缓存
        self._cache = {}
        self.cache_size = cache_size

    def _init_geocoding_chn(self):
        """初始化GeocodingCHN"""
        try:
            # 添加项目路径
            if self.data_path:
                sys.path.insert(0, str(Path(self.data_path).parent))

            # 尝试不同导入方式
            try:
                from GeocodingCHN.Geocoding import GeoCoder
                print("成功导入GeocodingCHN")
                return GeoCoder(self.data_path)
            except ImportError:
                try:
                    import GeocodingCHN
                    print("成功导入GeocodingCHN（单文件）")
                    return GeocodingCHN.GeoCoder(self.data_path)
                except ImportError:
                    print("警告: 无法导入GeocodingCHN，将使用模拟模式")
                    return self._create_mock_geocoder()

        except Exception as e:
            print(f"初始化GeocodingCHN失败: {e}")
            return self._create_mock_geocoder()

    def _create_mock_geocoder(self):
        """创建模拟地理编码器"""

        class MockGeocoder:
            def geocode(self, address):
                return None

            def reverse_geocode(self, lat, lon):
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
            result = self.geocoder.geocode(address)

            if result and 'lat' in result and 'lng' in result:
                parsed = {
                    'latitude': float(result['lat']),
                    'longitude': float(result['lng']),
                    'formatted_address': result.get('formatted_address', ''),
                    'province': result.get('province', ''),
                    'city': result.get('city', ''),
                    'district': result.get('district', ''),
                    'street': result.get('street', ''),
                    'street_number': result.get('street_number', ''),
                    'confidence': float(result.get('confidence', 0.0)),
                    'source': 'GeocodingCHN',
                    'timestamp': time.time()
                }

                self.stats['successful'] += 1
                response_time = time.time() - start_time
                self.stats['avg_response_time'] = (
                                                          self.stats['avg_response_time'] * (
                                                              self.stats['successful'] - 1) + response_time
                                                  ) / self.stats['successful']

                return parsed

        except Exception as e:
            print(f"地理编码失败: {address}, 错误: {e}")

        self.stats['failed'] += 1
        return None

    def geocode(self, address: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        地理编码：地址 -> 经纬度

        Args:
            address: 地址字符串
            use_cache: 是否使用缓存

        Returns:
            包含经纬度和解析信息的字典
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

    def reverse_geocode(self, latitude: float, longitude: float,
                        use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        逆地理编码：经纬度 -> 地址

        Args:
            latitude: 纬度
            longitude: 经度
            use_cache: 是否使用缓存

        Returns:
            地址信息字典
        """
        cache_key = f"reverse:{latitude:.6f},{longitude:.6f}"

        if use_cache and self.cache_enabled and cache_key in self._cache:
            self.stats['cache_hits'] += 1
            return self._cache[cache_key]

        try:
            result = self.geocoder.reverse_geocode(latitude, longitude)

            if result:
                parsed = {
                    'formatted_address': result.get('formatted_address', ''),
                    'province': result.get('province', ''),
                    'city': result.get('city', ''),
                    'district': result.get('district', ''),
                    'street': result.get('street', ''),
                    'street_number': result.get('street_number', ''),
                    'source': 'GeocodingCHN'
                }

                if use_cache and self.cache_enabled:
                    self._cache[cache_key] = parsed

                return parsed

        except Exception as e:
            print(f"逆地理编码失败: ({latitude}, {longitude}), 错误: {e}")

        return None

    def batch_geocode(self, addresses: List[str], n_jobs: int = 8,
                      use_cache: bool = True) -> List[Optional[Dict[str, Any]]]:
        """批量地理编码"""
        from joblib import Parallel, delayed

        def process_address(addr):
            return self.geocode(addr, use_cache)

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_address)(addr)
            for addr in addresses
        )

        return results

    def get_administrative_info(self, address: str) -> Dict[str, Any]:
        """获取行政区划信息"""
        geocode_result = self.geocode(address)

        if geocode_result:
            return {
                'province': geocode_result.get('province', ''),
                'city': geocode_result.get('city', ''),
                'district': geocode_result.get('district', ''),
                'street': geocode_result.get('street', ''),
                'country': '中国',
                'country_code': 'CN'
            }

        return {}

    def calculate_distance_haversine(self, lat1: float, lon1: float,
                                     lat2: float, lon2: float) -> float:
        """
        计算两个坐标之间的球面距离（公里）- Haversine公式

        Args:
            lat1, lon1: 第一个坐标
            lat2, lon2: 第二个坐标

        Returns:
            距离（公里）
        """
        # 将角度转换为弧度
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # Haversine公式
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # 地球平均半径，单位公里

        return c * r

    def calculate_distance_vincenty(self, lat1: float, lon1: float,
                                    lat2: float, lon2: float) -> float:
        """
        计算两个坐标之间的距离（公里）- Vincenty公式（更精确）
        """
        from math import atan2, cos, sin, sqrt, radians

        # 转换为弧度
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        # WGS-84椭球体参数
        a = 6378137.0  # 长半轴，单位米
        f = 1 / 298.257223563  # 扁率
        b = a * (1 - f)  # 短半轴

        # Vincenty公式
        L = lon2 - lon1
        U1 = atan2((1 - f) * sin(lat1), cos(lat1))
        U2 = atan2((1 - f) * sin(lat2), cos(lat2))
        sinU1 = sin(U1)
        cosU1 = cos(U1)
        sinU2 = sin(U2)
        cosU2 = cos(U2)

        lambda_val = L
        for _ in range(100):  # 迭代计算
            sin_lambda = sin(lambda_val)
            cos_lambda = cos(lambda_val)
            sin_sigma = sqrt((cosU2 * sin_lambda) ** 2 +
                             (cosU1 * sinU2 - sinU1 * cosU2 * cos_lambda) ** 2)
            if sin_sigma == 0:
                return 0.0

            cos_sigma = sinU1 * sinU2 + cosU1 * cosU2 * cos_lambda
            sigma = atan2(sin_sigma, cos_sigma)
            sin_alpha = cosU1 * cosU2 * sin_lambda / sin_sigma
            cos2_alpha = 1 - sin_alpha ** 2
            cos2_sigma_m = cos_sigma - 2 * sinU1 * sinU2 / cos2_alpha

            C = f / 16 * cos2_alpha * (4 + f * (4 - 3 * cos2_alpha))
            lambda_prev = lambda_val
            lambda_val = L + (1 - C) * f * sin_alpha * (
                    sigma + C * sin_sigma * (
                    cos2_sigma_m + C * cos_sigma * (-1 + 2 * cos2_sigma_m ** 2)
            )
            )

            if abs(lambda_val - lambda_prev) < 1e-12:
                break

        u2 = cos2_alpha * (a ** 2 - b ** 2) / (b ** 2)
        A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
        B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
        delta_sigma = B * sin_sigma * (
                cos2_sigma_m + B / 4 * (
                cos_sigma * (-1 + 2 * cos2_sigma_m ** 2) -
                B / 6 * cos2_sigma_m * (-3 + 4 * sin_sigma ** 2) *
                (-3 + 4 * cos2_sigma_m ** 2)
        )
        )

        distance = b * A * (sigma - delta_sigma)  # 单位米
        return distance / 1000  # 转换为公里

    def calculate_distance(self, lat1: float, lon1: float,
                           lat2: float, lon2: float, method: str = 'haversine') -> float:
        """
        计算两个坐标之间的距离

        Args:
            lat1, lon1: 第一个坐标
            lat2, lon2: 第二个坐标
            method: 计算方法（haversine或vincenty）

        Returns:
            距离（公里）
        """
        if method == 'vincenty':
            return self.calculate_distance_vincenty(lat1, lon1, lat2, lon2)
        else:
            return self.calculate_distance_haversine(lat1, lon1, lat2, lon2)

    def find_nearby_addresses(self, center_lat: float, center_lon: float,
                              addresses: List[Dict[str, Any]], radius_km: float = 5) -> List[Dict[str, Any]]:
        """
        查找附近地址

        Args:
            center_lat: 中心点纬度
            center_lon: 中心点经度
            addresses: 地址列表，每个元素需包含latitude和longitude
            radius_km: 搜索半径（公里）

        Returns:
            附近地址列表
        """
        nearby = []

        for addr in addresses:
            if 'latitude' in addr and 'longitude' in addr:
                distance = self.calculate_distance(
                    center_lat, center_lon,
                    addr['latitude'], addr['longitude']
                )

                if distance <= radius_km:
                    addr_copy = addr.copy()
                    addr_copy['distance_km'] = distance
                    nearby.append(addr_copy)

        # 按距离排序
        nearby.sort(key=lambda x: x['distance_km'])

        return nearby

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
        print("缓存已清空")

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

            print(f"缓存已保存到: {filepath}")

        except Exception as e:
            print(f"保存缓存失败: {e}")

    def load_cache_from_file(self, filepath: str):
        """从文件加载缓存"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    loaded_cache = json.load(f)

                self._cache.update(loaded_cache)
                print(f"从 {filepath} 加载了 {len(loaded_cache)} 个缓存项")

        except Exception as e:
            print(f"加载缓存失败: {e}")


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

        # 可以添加其他数据源
        if 'amap' in self.api_keys:
            self.sources['amap'] = self._init_amap()

        if 'baidu' in self.api_keys:
            self.sources['baidu'] = self._init_baidu()

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
                    print(f"数据源 {source} 失败: {e}")

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