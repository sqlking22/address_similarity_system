#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2025/12/5 15:15
# @Author  : hejun
"""
地址聚类模块
"""
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from typing import Dict, List, Any, Optional
import pandas as pd
from .geocoding_integration import GeocodingCHNIntegration  # 使用相对导入
from utils.logger import setup_logging
from .similarity_calculator import calculate_haversine_distance  # 导入已有的距离计算函数

# 初始化日志记录器
logger = setup_logging('clustering.py').get_logger()


class AddressClustering:
    """地址聚类器"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # 聚类参数
        self.clustering_config = self.config.get('clustering', {
            'similarity_threshold': 0.7,
            'min_cluster_size': 2,
            'max_cluster_radius_km': 5
        })

        # 聚类算法
        self.algorithm = self.config.get('algorithm', 'connected_components')

    def cluster_by_connected_components(self,
                                        similarity_matrix: np.ndarray,
                                        threshold: float = None) -> np.ndarray:
        """
        通过连通分量聚类

        Args:
            similarity_matrix: 相似度矩阵
            threshold: 相似度阈值

        Returns:
            聚类标签数组
        """
        if threshold is None:
            threshold = self.clustering_config['similarity_threshold']

        n = len(similarity_matrix)
        adj_matrix = np.zeros((n, n), dtype=bool)

        # 更严格的相似度判断
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i, j] >= threshold:
                    # 可以添加额外约束，例如行政区划必须相同等
                    adj_matrix[i, j] = True
                    adj_matrix[j, i] = True

        adj_sparse = csr_matrix(adj_matrix)
        n_components, labels = connected_components(
            csgraph=adj_sparse,
            directed=False,
            return_labels=True
        )

        return labels

    def cluster_with_spatial_constraints(self,
                                         similarity_matrix: np.ndarray,
                                         coordinates: np.ndarray,
                                         similarity_threshold: float = None,
                                         spatial_threshold_km: float = None) -> np.ndarray:
        """
        带空间约束的聚类

        Args:
            similarity_matrix: 相似度矩阵
            coordinates: 坐标数组 (n, 2) -> [longitude, latitude]
            similarity_threshold: 相似度阈值
            spatial_threshold_km: 空间距离阈值

        Returns:
            聚类标签数组
        """
        if similarity_threshold is None:
            similarity_threshold = self.clustering_config['similarity_threshold']

        if spatial_threshold_km is None:
            spatial_threshold_km = self.clustering_config['max_cluster_radius_km']

        n = len(similarity_matrix)

        # 创建带空间约束的邻接矩阵
        adj_matrix = lil_matrix((n, n), dtype=bool)

        for i in range(n):
            for j in range(i + 1, n):
                # 文本相似度条件
                if similarity_matrix[i, j] >= similarity_threshold:
                    # 空间距离条件
                    if coordinates[i] is not None and coordinates[j] is not None:
                        # 计算空间距离，使用已有的距离计算函数
                        lat1, lon1 = coordinates[i]
                        lat2, lon2 = coordinates[j]

                        distance_km = calculate_haversine_distance(lat1, lon1, lat2, lon2)

                        if distance_km <= spatial_threshold_km:
                            adj_matrix[i, j] = True
                            adj_matrix[j, i] = True
                    else:
                        # 如果没有坐标，只依赖文本相似度
                        adj_matrix[i, j] = True
                        adj_matrix[j, i] = True

        # 转换为CSR格式
        adj_sparse = adj_matrix.tocsr()

        # 计算连通分量
        n_components, labels = connected_components(
            csgraph=adj_sparse,
            directed=False,
            return_labels=True
        )

        return labels

    def cluster_with_dbscan(self,
                            similarity_matrix: np.ndarray,
                            coordinates: Optional[np.ndarray] = None,
                            eps: float = None,
                            min_samples: int = None) -> np.ndarray:
        """
        使用DBSCAN聚类

        Args:
            similarity_matrix: 相似度矩阵
            coordinates: 坐标数组（可选）
            eps: 邻域半径
            min_samples: 最小样本数

        Returns:
            聚类标签数组
        """

        # 从配置获取默认值
        if eps is None:
            eps = self.config.get('clustering_algorithms', {}).get('dbscan', {}).get('eps', 0.5)
        if min_samples is None:
            min_samples = self.config.get('clustering_algorithms', {}).get('dbscan', {}).get('min_samples', 2)

        # 转换相似度为距离
        distance_matrix = 1 - similarity_matrix

        # 如果有坐标，可以结合空间距离
        if coordinates is not None and len(coordinates) > 0:
            # 创建特征矩阵：文本距离 + 空间距离
            n = len(distance_matrix)
            features = []

            for i in range(n):
                feature = []
                # 添加文本相似度特征
                feature.extend(distance_matrix[i])

                # 添加坐标特征（归一化）
                if coordinates[i] is not None:
                    lat, lon = coordinates[i]
                    # 简单归一化
                    lat_norm = (lat + 90) / 180  # [-90, 90] -> [0, 1]
                    lon_norm = (lon + 180) / 360  # [-180, 180] -> [0, 1]
                    feature.extend([lat_norm, lon_norm])
                else:
                    feature.extend([0, 0])

                features.append(feature)

            X = np.array(features)
        else:
            X = distance_matrix

        # DBSCAN聚类
        clustering = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='precomputed' if coordinates is None else 'euclidean',
            n_jobs=-1
        ).fit(X)

        return clustering.labels_

    def hierarchical_clustering(self,
                                similarity_matrix: np.ndarray,
                                n_clusters: Optional[int] = None,
                                distance_threshold: float = None) -> np.ndarray:
        """
        层次聚类

        Args:
            similarity_matrix: 相似度矩阵
            n_clusters: 聚类数量
            distance_threshold: 距离阈值

        Returns:
            聚类标签数组
        """
        # 从配置获取链接方式
        linkage = self.config.get('clustering_algorithms', {}).get('hierarchical', {}).get('linkage', 'average')

        # 转换相似度为距离
        distance_matrix = 1 - similarity_matrix

        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            distance_threshold=distance_threshold,
            metric='precomputed',
            linkage=linkage
        )

        labels = clustering.fit_predict(distance_matrix)

        return labels

    def merge_small_clusters(self, labels: np.ndarray,
                             min_size: int = 2) -> np.ndarray:
        """
        合并小聚类

        Args:
            labels: 原始聚类标签
            min_size: 最小聚类大小

        Returns:
            合并后的聚类标签
        """
        from collections import Counter
        label_counts = Counter(labels)

        # 找出小聚类（排除噪声点-1）
        small_clusters = [
            label for label, count in label_counts.items()
            if count < min_size and label != -1
        ]

        if not small_clusters:
            return labels

        # 找出最大的聚类
        valid_clusters = {k: v for k, v in label_counts.items() if k != -1}
        if valid_clusters:
            largest_cluster = max(valid_clusters.items(), key=lambda x: x[1])[0]
        else:
            largest_cluster = 0

        # 合并小聚类到最大聚类
        new_labels = labels.copy()
        for label in small_clusters:
            new_labels[new_labels == label] = largest_cluster

        return new_labels

    def optimize_clustering(self, labels: np.ndarray,
                            similarity_matrix: np.ndarray,
                            min_similarity: float = None) -> np.ndarray:
        """
        优化聚类结果

        Args:
            labels: 原始聚类标签
            similarity_matrix: 相似度矩阵
            min_similarity: 最小相似度

        Returns:
            优化后的聚类标签
        """
        if min_similarity is None:
            min_similarity = self.config.get('clustering_algorithms', {}).get('optimization', {}).get('min_similarity',
                                                                                                      0.5)

        n = len(labels)
        new_labels = labels.copy()

        # 重新分配相似度高的点
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i, j] >= min_similarity:
                    # 如果两个点高度相似但不在同一个聚类，合并它们
                    if new_labels[i] != new_labels[j]:
                        # 合并到较小的聚类标签
                        target_label = min(new_labels[i], new_labels[j])
                        source_label = max(new_labels[i], new_labels[j])
                        new_labels[new_labels == source_label] = target_label

        return new_labels

    def validate_clusters(self, labels: np.ndarray,
                          address_data: Dict[int, Dict[str, Any]],
                          similarity_matrix: np.ndarray) -> np.ndarray:
        """
        验证聚类结果，拆分不合理的聚类

        Args:
            labels: 聚类标签数组
            address_data: 地址数据字典
            similarity_matrix: 相似度矩阵

        Returns:
            验证后的聚类标签数组
        """
        new_labels = labels.copy()

        # 对每个聚类组进行验证
        unique_labels = np.unique(labels)
        next_label = max(unique_labels) + 1 if len(unique_labels) > 0 else 0

        for label in unique_labels:
            if label == -1:  # 噪声点跳过
                continue

            # 获取该聚类的所有地址索引
            indices = np.where(labels == label)[0]

            # 如果聚类中有明显不相关的地址，进行拆分
            if len(indices) > 2:
                # 检查聚类内地址的相似度是否合理
                needs_split = False

                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        idx1, idx2 = indices[i], indices[j]
                        # 如果任意两个地址相似度很低，则需要拆分
                        if similarity_matrix[idx1, idx2] < 0.6:  # 阈值可以根据需要调整
                            needs_split = True
                            break
                    if needs_split:
                        break

                # 如果需要拆分，使用层次聚类重新分组
                if needs_split:
                    # 构建该聚类的子相似度矩阵
                    sub_similarity_matrix = similarity_matrix[np.ix_(indices, indices)]

                    # 简单的拆分策略：基于连通分量
                    adj_matrix = sub_similarity_matrix >= 0.6
                    np.fill_diagonal(adj_matrix, True)

                    # 计算连通分量
                    from scipy.sparse import csr_matrix
                    from scipy.sparse.csgraph import connected_components

                    adj_sparse = csr_matrix(adj_matrix)
                    n_components, sub_labels = connected_components(
                        csgraph=adj_sparse,
                        directed=False,
                        return_labels=True
                    )

                    # 如果拆分成多个组件，则重新分配标签
                    if n_components > 1:
                        logger.info(f"拆分聚类 {label} 为 {n_components} 个子聚类")
                        # 保持主组件的标签，为其他组件分配新标签
                        main_component_label = sub_labels[0]  # 假设第一个是主组件

                        for i, idx in enumerate(indices):
                            if sub_labels[i] != main_component_label:
                                new_labels[idx] = next_label + sub_labels[i]

                        # 更新下一个可用标签
                        next_label += n_components - 1

        return new_labels

    def calculate_cluster_quality(self, labels: np.ndarray,
                                  similarity_matrix: np.ndarray) -> Dict[str, float]:
        """
        计算聚类质量指标

        Args:
            labels: 聚类标签
            similarity_matrix: 相似度矩阵

        Returns:
            质量指标字典
        """
        sampling_size = self.config.get('clustering_algorithms', {}).get('optimization', {}).get('sampling_size', 5)

        n = len(labels)
        unique_labels = set(labels)

        if -1 in unique_labels:  # 排除噪声点
            unique_labels.remove(-1)

        if len(unique_labels) == 0:
            return {
                'n_clusters': 0,
                'avg_similarity_within': 0,
                'avg_similarity_between': 0,
                'silhouette_score': -1
            }

        # 计算类内相似度
        within_similarities = []
        for label in unique_labels:
            indices = np.where(labels == label)[0]
            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        idx1, idx2 = indices[i], indices[j]
                        within_similarities.append(similarity_matrix[idx1, idx2])

        # 计算类间相似度
        between_similarities = []
        label_list = list(unique_labels)
        for i in range(len(label_list)):
            for j in range(i + 1, len(label_list)):
                indices1 = np.where(labels == label_list[i])[0]
                indices2 = np.where(labels == label_list[j])[0]

                for idx1 in indices1[:sampling_size]:  # 采样，避免计算量过大
                    for idx2 in indices2[:sampling_size]:
                        between_similarities.append(similarity_matrix[idx1, idx2])

        # 计算轮廓系数（简化版）
        silhouette_sum = 0
        valid_points = 0

        for i in range(n):
            if labels[i] == -1:
                continue

            # 找到同类的其他点
            same_cluster_indices = np.where(labels == labels[i])[0]
            same_cluster_indices = same_cluster_indices[same_cluster_indices != i]

            if len(same_cluster_indices) == 0:
                continue

            # 计算a(i)：i到同簇其他点的平均距离
            a_i = 1 - np.mean([similarity_matrix[i, j] for j in same_cluster_indices])

            # 计算b(i)：i到其他簇的最小平均距离
            other_labels = [l for l in unique_labels if l != labels[i]]
            b_i_values = []

            for other_label in other_labels:
                other_cluster_indices = np.where(labels == other_label)[0]
                if len(other_cluster_indices) > 0:
                    avg_similarity = np.mean([similarity_matrix[i, j]
                                              for j in other_cluster_indices[:sampling_size]])  # 采样
                    b_i_values.append(1 - avg_similarity)

            if b_i_values:
                b_i = min(b_i_values)
                silhouette_i = (b_i - a_i) / max(a_i, b_i)
                silhouette_sum += silhouette_i
                valid_points += 1

        avg_silhouette = silhouette_sum / valid_points if valid_points > 0 else -1

        return {
            'n_clusters': len(unique_labels),
            'avg_similarity_within': np.mean(within_similarities) if within_similarities else 0,
            'avg_similarity_between': np.mean(between_similarities) if between_similarities else 0,
            'silhouette_score': avg_silhouette,
            'within_between_ratio': (
                np.mean(within_similarities) / np.mean(between_similarities)
                if within_similarities and between_similarities and np.mean(between_similarities) > 0
                else 0
            )
        }

    def create_cluster_summary(self,
                               labels: np.ndarray,
                               address_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        创建聚类摘要

        Args:
            labels: 聚类标签
            address_data: 地址数据列表

        Returns:
            聚类摘要DataFrame
        """
        from collections import defaultdict

        # 按聚类分组
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[label].append(idx)

        # 生成摘要
        summaries = []

        for cluster_id, indices in clusters.items():
            if cluster_id == -1:  # 噪声点
                continue

            cluster_addresses = [address_data[idx] for idx in indices]

            # 计算聚类统计
            cluster_size = len(cluster_addresses)

            # 提取代表性地址（最常见的标准化地址）
            standardized_addrs = [addr.get('standardized', '') for addr in cluster_addresses]
            if standardized_addrs:
                from collections import Counter
                addr_counter = Counter(standardized_addrs)
                representative = addr_counter.most_common(1)[0][0]
            else:
                representative = ''

            # 计算平均坐标（如果有）
            latitudes = [addr.get('latitude') for addr in cluster_addresses
                         if addr.get('latitude') is not None]
            longitudes = [addr.get('longitude') for addr in cluster_addresses
                          if addr.get('longitude') is not None]

            avg_lat = np.mean(latitudes) if latitudes else None
            avg_lon = np.mean(longitudes) if longitudes else None

            # 提取示例地址（前3个）
            examples = [addr.get('original', '')[:50] for addr in cluster_addresses[:3]]

            summaries.append({
                'cluster_id': int(cluster_id),
                'size': cluster_size,
                'representative_address': representative[:100],
                'avg_latitude': avg_lat,
                'avg_longitude': avg_lon,
                'examples': examples,
                'member_ids': indices
            })

        # 按大小排序
        summaries.sort(key=lambda x: x['size'], reverse=True)

        return pd.DataFrame(summaries)
