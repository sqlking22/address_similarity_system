#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2025/12/5 15:16
# @Author  : hejun
"""
多维度相似度计算模块
结合文本、空间和行政区划相似度
"""
import numpy as np
from numba import jit, prange
from rapidfuzz import fuzz
import Levenshtein
from jaro import jaro_winkler_metric
from datasketch import MinHash, MinHashLSH
from typing import Dict, Any, List, Tuple, Optional
import math
import collections


def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    计算两个坐标之间的球面距离（公里）- Haversine公式

    Args:
        lat1, lon1: 第一个坐标
        lat2, lon2: 第二个坐标

    Returns:
        距离（公里）
    """
    # 将角度转换为弧度
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine公式
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distance_km = 6371 * c  # 地球平均半径，单位公里

    return distance_km


class LRUCache:
    """LRU缓存实现，限制缓存大小"""

    def __init__(self, maxsize: int = 10000):
        self.cache = collections.OrderedDict()
        self.maxsize = maxsize

    def get(self, key):
        if key in self.cache:
            # 将访问的键移到末尾（标记为最近使用）
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            # 更新现有键值
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.maxsize:
            # 删除最久未使用的项（第一个项）
            self.cache.popitem(last=False)

        self.cache[key] = value


class MultiDimensionalSimilarityCalculator:
    """多维度相似度计算器"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # 权重配置
        self.weights = self.config.get('weights', {
            'text': 0.5,
            'spatial': 0.3,
            'admin': 0.2
        })

        # 文本相似度参数
        self.text_config = self.config.get('text_similarity', {
            'minhash_permutations': 256,
            'ngram_size': 3,
            'lsh_threshold': 0.3
        })

        # 空间相似度参数
        self.spatial_config = self.config.get('spatial_similarity', {
            'distance_threshold_km': 10,
            'max_search_radius_km': 50
        })

        # 初始化MinHash缓存（使用LRU缓存）
        self.minhash_cache = LRUCache(maxsize=10000)

    def calculate_text_similarity_batch(self, text1_list: List[str], text2_list: List[str]) -> np.ndarray:
        """
        批量计算文本相似度（向量化）
        """
        n = len(text1_list)
        scores = np.zeros(n)

        for i in range(n):
            scores[i] = self.calculate_text_similarity_single(
                text1_list[i], text2_list[i]
            )

        return scores

    def calculate_text_similarity_single(self, text1: str, text2: str,
                                         method: str = 'combined') -> float:
        """
        计算单个文本相似度

        Args:
            text1: 文本1
            text2: 文本2
            method: 计算方法（combined, minhash, edit, jaro）

        Returns:
            相似度得分（0-1）
        """
        if not text1 or not text2:
            return 0.0

        if method == 'minhash':
            return self._text_similarity_minhash(text1, text2)
        elif method == 'edit':
            return self._text_similarity_edit(text1, text2)
        elif method == 'jaro':
            return self._text_similarity_jaro(text1, text2)
        else:  # combined
            return self._text_similarity_combined(text1, text2)

    def _text_similarity_minhash(self, text1: str, text2: str) -> float:
        """MinHash相似度"""
        # 生成或获取MinHash
        m1 = self._get_minhash(text1)
        m2 = self._get_minhash(text2)

        return m1.jaccard(m2)

    def _get_minhash(self, text: str) -> MinHash:
        """获取文本的MinHash"""
        cached = self.minhash_cache.get(text)
        if cached is not None:
            return cached

        m = MinHash(num_perm=self.text_config['minhash_permutations'])

        # 生成n-gram
        n = self.text_config['ngram_size']
        for i in range(len(text) - n + 1):
            ngram = text[i:i + n]
            m.update(ngram.encode('utf-8'))

        self.minhash_cache.put(text, m)
        return m

    def _text_similarity_edit(self, text1: str, text2: str) -> float:
        """编辑距离相似度"""
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 0.0

        edit_dist = Levenshtein.distance(text1, text2)
        return 1.0 - (edit_dist / max_len)

    def _text_similarity_jaro(self, text1: str, text2: str) -> float:
        """Jaro-Winkler相似度"""
        return jaro_winkler_metric(text1, text2)

    def _text_similarity_combined(self, text1: str, text2: str) -> float:
        """综合文本相似度"""
        # Jaro-Winkler（前缀敏感）
        jaro_score = jaro_winkler_metric(text1, text2)

        # 归一化编辑距离
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            edit_score = 0.0
        else:
            edit_dist = Levenshtein.distance(text1, text2)
            edit_score = 1.0 - (edit_dist / max_len)

        # 集合相似度（Jaccard）
        set1 = set(text1)
        set2 = set(text2)
        if set1 or set2:
            jaccard_score = len(set1 & set2) / len(set1 | set2)
        else:
            jaccard_score = 0.0

        # 加权综合
        combined_score = 0.4 * jaro_score + 0.3 * edit_score + 0.3 * jaccard_score

        return min(1.0, max(0.0, combined_score))

    @staticmethod
    @jit(nopython=True, parallel=True)
    def calculate_spatial_similarity_batch(lat1_array: np.ndarray, lon1_array: np.ndarray,
                                           lat2_array: np.ndarray, lon2_array: np.ndarray,
                                           threshold_km: float = 10) -> np.ndarray:
        """
        批量计算空间相似度（Numba加速）
        """
        n = len(lat1_array)
        similarities = np.zeros(n)

        for i in prange(n):
            lat1, lon1, lat2, lon2 = lat1_array[i], lon1_array[i], lat2_array[i], lon2_array[i]

            # 检查是否有效坐标
            if np.isnan(lat1) or np.isnan(lon1) or np.isnan(lat2) or np.isnan(lon2):
                similarities[i] = 0.0
                continue

            # 计算Haversine距离
            lat1_rad, lon1_rad, lat2_rad, lon2_rad = (
                np.radians(lat1), np.radians(lon1),
                np.radians(lat2), np.radians(lon2)
            )

            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
            c = 2 * np.arcsin(np.sqrt(a))
            distance_km = 6371 * c

            # 转换为相似度（指数衰减）
            similarities[i] = np.exp(-distance_km / threshold_km)

        return similarities

    def calculate_spatial_similarity_single(self, lat1: float, lon1: float,
                                            lat2: float, lon2: float) -> float:
        """
        计算单个空间相似度
        """
        # 处理缺失值
        if (np.isnan(lat1) or np.isnan(lon1) or
                np.isnan(lat2) or np.isnan(lon2)):
            return 0.0

        # 计算距离（使用本地函数避免循环导入）
        distance_km = calculate_haversine_distance(lat1, lon1, lat2, lon2)

        # 转换为相似度
        threshold = self.spatial_config['distance_threshold_km']
        similarity = np.exp(-distance_km / threshold)

        return float(similarity)

    def calculate_administrative_similarity(self, comp1: Dict[str, Any],
                                            comp2: Dict[str, Any]) -> float:
        """
        计算行政区划相似度
        """
        similarity = 0.0

        # 1. 省份匹配（最重要）
        prov1 = comp1.get('province', '')
        prov2 = comp2.get('province', '')
        if prov1 and prov2:
            if prov1 == prov2:
                similarity += 0.4
            else:
                # 省份不同，相似度大幅降低
                return 0.1

        # 2. 城市匹配
        city1 = comp1.get('city', '')
        city2 = comp2.get('city', '')
        if city1 and city2:
            if city1 == city2:
                similarity += 0.3
            elif prov1 == prov2:  # 同省不同市
                similarity += 0.1

        # 3. 区县匹配
        district1 = comp1.get('district', '')
        district2 = comp2.get('district', '')
        if district1 and district2:
            if district1 == district2:
                similarity += 0.2
            elif city1 == city2:  # 同市不同区
                similarity += 0.1

        # 4. 街道匹配
        street1 = comp1.get('street', '')
        street2 = comp2.get('street', '')
        if street1 and street2:
            if street1 == street2:
                similarity += 0.1

        return min(1.0, similarity)

    def calculate_comprehensive_similarity(self,
                                           text1: str, text2: str,
                                           lat1: Optional[float], lon1: Optional[float],
                                           lat2: Optional[float], lon2: Optional[float],
                                           comp1: Dict[str, Any], comp2: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算综合相似度

        Returns:
            包含各个维度相似度和综合相似度的字典
        """
        # 计算各维度相似度
        text_sim = self.calculate_text_similarity_single(text1, text2)
        spatial_sim = self.calculate_spatial_similarity_single(lat1, lon1, lat2, lon2)
        admin_sim = self.calculate_administrative_similarity(comp1, comp2)

        # 计算综合相似度
        comprehensive_sim = (
                self.weights['text'] * text_sim +
                self.weights['spatial'] * spatial_sim +
                self.weights['admin'] * admin_sim
        )

        # 地理距离
        distance_km = None
        if (lat1 is not None and lon1 is not None and
                lat2 is not None and lon2 is not None):
            distance_km = calculate_haversine_distance(lat1, lon1, lat2, lon2)

        return {
            'comprehensive_similarity': min(1.0, max(0.0, comprehensive_sim)),
            'text_similarity': text_sim,
            'spatial_similarity': spatial_sim,
            'administrative_similarity': admin_sim,
            'distance_km': distance_km,
            'is_similar': comprehensive_sim >= self.config.get('similarity_threshold', 0.7),
            'weights': self.weights.copy()
        }

    def batch_calculate_similarities(self,
                                     address_pairs: List[Tuple[int, int, Dict[str, Any]]],
                                     address_data: Dict[int, Dict[str, Any]],
                                     n_jobs: int = 8) -> List[Dict[str, Any]]:
        """
        批量计算相似度

        Args:
            address_pairs: 地址对列表，每个元素为(id1, id2, 额外信息)
            address_data: 地址数据字典，key为ID，value为地址信息
            n_jobs: 并行任务数

        Returns:
            相似度结果列表
        """
        from joblib import Parallel, delayed

        def process_pair(pair):
            id1, id2, extra_info = pair

            if id1 not in address_data or id2 not in address_data:
                return None

            addr1 = address_data[id1]
            addr2 = address_data[id2]

            similarity = self.calculate_comprehensive_similarity(
                text1=addr1.get('standardized', ''),
                text2=addr2.get('standardized', ''),
                lat1=addr1.get('latitude'),
                lon1=addr1.get('longitude'),
                lat2=addr2.get('latitude'),
                lon2=addr2.get('longitude'),
                comp1=addr1.get('parsed', {}),
                comp2=addr2.get('parsed', {})
            )

            similarity.update({
                'id1': id1,
                'id2': id2,
                'address1': addr1.get('original', ''),
                'address2': addr2.get('original', ''),
                **extra_info
            })

            return similarity

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_pair)(pair)
            for pair in address_pairs
        )

        # 过滤None结果
        return [r for r in results if r is not None]

    def find_similar_candidates_with_lsh(self,
                                         address_data: Dict[int, Dict[str, Any]],
                                         threshold: float = 0.3) -> List[Tuple[int, int]]:
        """
        使用LSH查找相似候选对

        Args:
            address_data: 地址数据字典
            threshold: LSH阈值

        Returns:
            候选对列表
        """
        # 创建LSH索引
        lsh = MinHashLSH(
            threshold=threshold,
            num_perm=self.text_config['minhash_permutations']
        )

        # 插入所有地址的MinHash
        for addr_id, addr_info in address_data.items():
            text = addr_info.get('standardized', '')
            if text:
                mh = self._get_minhash(text)
                lsh.insert(str(addr_id), mh)

        # 查找候选对
        candidate_pairs = set()

        for addr_id, addr_info in address_data.items():
            text = addr_info.get('standardized', '')
            if text:
                mh = self._get_minhash(text)
                candidates = lsh.query(mh)

                # 添加候选对（排除自身）
                for cand_id_str in candidates:
                    cand_id = int(cand_id_str)
                    if cand_id != addr_id:
                        pair = (min(addr_id, cand_id), max(addr_id, cand_id))
                        candidate_pairs.add(pair)

        return list(candidate_pairs)

    def create_similarity_matrix(self,
                                 address_ids: List[int],
                                 address_data: Dict[int, Dict[str, Any]],
                                 similarity_pairs: List[Dict[str, Any]]) -> np.ndarray:
        """
        创建相似度矩阵

        Args:
            address_ids: 地址ID列表
            address_data: 地址数据
            similarity_pairs: 相似度对列表

        Returns:
            相似度矩阵
        """
        n = len(address_ids)
        sim_matrix = np.zeros((n, n))

        # 创建ID到索引的映射
        id_to_idx = {addr_id: idx for idx, addr_id in enumerate(address_ids)}

        # 填充相似度矩阵
        for pair in similarity_pairs:
            idx1 = id_to_idx.get(pair['id1'])
            idx2 = id_to_idx.get(pair['id2'])

            if idx1 is not None and idx2 is not None:
                sim = pair['comprehensive_similarity']
                sim_matrix[idx1, idx2] = sim
                sim_matrix[idx2, idx1] = sim

        # 对角线设为1（自身相似度）
        np.fill_diagonal(sim_matrix, 1.0)

        return sim_matrix
