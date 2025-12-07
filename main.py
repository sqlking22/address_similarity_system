# main.py
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# ****************************************************************#
# @Time    : 2025/12/6 0:11
# @Author  : JonHe
# Function :
# ****************************************************************#
"""
基于数据库的地址相似度识别主程序
支持从MySQL读取数据并存储结果
"""
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import time
import argparse
import json
# 进度条
from tqdm import tqdm
from datetime import datetime
from utils.file_utils import FileUtils
from core.address_normalizer import AdvancedAddressNormalizer
from core.geocoding_integration import GeocodingCHNIntegration
from core.clustering import AddressClustering
from core.similarity_calculator import MultiDimensionalSimilarityCalculator
from utils.parallel_processor import ParallelProcessor
from utils.db_handler import DatabaseHandler
from config.config import Config
from utils.parallel_processor import SimilarityMatrixBuilder
from utils.logger import setup_logging

# 初始化日志记录器
logger = setup_logging('main.py').get_logger()


class AddressSimilaritySystemDB:
    """基于数据库的地址相似度识别系统"""

    def __init__(self, ):
        self.db_handler = DatabaseHandler(Config.DATABASE_CONFIG)
        self.normalizer = AdvancedAddressNormalizer(Config.ALGORITHM_CONFIG.get('normalization', {}))
        self.geocoder = GeocodingCHNIntegration(
            cache_size=10000
        )
        self.clustering = AddressClustering(Config.ALGORITHM_CONFIG)
        self.similarity_calculator = MultiDimensionalSimilarityCalculator(Config.ALGORITHM_CONFIG)
        self.processor = ParallelProcessor(
            n_jobs=Config.ALGORITHM_CONFIG.get('performance', {}).get('n_jobs', 8)
        )

    def connect_database(self):
        """连接数据库"""
        self.db_handler.connect()

    # 在处理前备份重要数据
    def backup_processing_data(self, data_snapshot):
        """备份处理数据"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = Config.DATA_DIR / f"backup_{timestamp}.json"
        FileUtils.write_json(data_snapshot, backup_file)

    def process_incremental_addresses(self, source_table: str, result_table: str,
                                      last_processed_time: datetime = None):
        """
        增量处理地址数据

        Args:
            source_table: 源数据表名
            result_table: 结果表名
            last_processed_time: 上次处理时间
        """
        logger.info("开始增量处理地址数据...")

        # 1. 读取新增地址数据
        start_time = time.time()
        new_address_df = self.db_handler.fetch_new_addresses(source_table, last_processed_time)

        if new_address_df.empty:
            logger.info("没有找到新增地址数据")
            return

        logger.info(f"读取新增数据耗时: {time.time() - start_time:.2f}秒")

        # 2. 标准化新增地址
        logger.info("开始标准化新增地址...")
        start_time = time.time()

        new_addresses = new_address_df['original_address'].fillna('').astype(str).tolist()
        new_normalized_results = self.normalizer.batch_normalize(
            new_addresses,
            n_jobs=Config.ALGORITHM_CONFIG.get('performance', {}).get('n_jobs', 8)
        )

        logger.info(f"新增地址标准化完成，耗时: {time.time() - start_time:.2f}秒")

        # 3. 地理编码新增地址
        logger.info("执行新增地址地理编码...")
        start_time = time.time()

        new_standardized_addresses = [
            result['standardized'] for result in new_normalized_results
        ]

        new_geocoded_results = self.processor.batch_process(
            new_standardized_addresses,
            self.geocoder.geocode,
            batch_size=1000,
            desc="新增地址地理编码"
        )

        logger.info(f"新增地址地理编码完成，耗时: {time.time() - start_time:.2f}秒")

        # 4. 准备新增地址数据
        new_address_data = {}
        for i, (idx, row) in enumerate(new_address_df.iterrows()):
            norm_result = new_normalized_results[i]
            geo_result = new_geocoded_results[i]

            # 从 offline_geocode 中提取坐标
            latitude, longitude = None, None
            if 'offline_geocode' in row and row['offline_geocode']:
                latitude, longitude = self.extract_coordinates_from_offline_geocode(row['offline_geocode'])

            # 如果 offline_geocode 中没有有效坐标，则使用 geocoded_results 中的结果
            if latitude is None or longitude is None:
                if geo_result:
                    latitude = geo_result.get('latitude')
                    longitude = geo_result.get('longitude')

            new_address_data[row['id']] = {
                'id': row['id'],
                'original': norm_result.get('original', ''),
                'standardized': norm_result.get('standardized', ''),
                'cleaned': norm_result.get('cleaned', norm_result.get('original', '')),  # 确保有默认值
                'corrected': norm_result.get('corrected', norm_result.get('standardized', '')),  # 确保有默认值
                'latitude': latitude,
                'longitude': longitude,
                'parsed': norm_result.get('parsed', {}),
                'components': norm_result.get('components', {}),
                'geo_info': geo_result or {}
            }

        # 5. 获取现有聚类结果
        logger.info("获取现有聚类结果...")
        existing_clusters_df = self.db_handler.fetch_cluster_representatives(result_table)

        if existing_clusters_df.empty:
            logger.info("没有现有聚类结果，执行全量聚类...")
            # 如果没有现有聚类，执行全量处理
            self._process_full_clustering_for_new_addresses(new_address_data, result_table)
            return

        # 6. 将新增地址与现有聚类进行匹配
        logger.info("匹配新增地址到现有聚类...")
        matched_results = self._match_new_addresses_to_existing_clusters(new_address_data, existing_clusters_df)

        # 7. 保存匹配结果到数据库
        self._save_incremental_results(matched_results, new_address_data, result_table)

        logger.info("增量地址识别处理完成！")

    def _process_full_clustering_for_new_addresses(self, address_data: Dict[int, Dict[str, Any]], result_table: str):
        """
        对新增地址执行全量聚类处理

        Args:
            address_data: 新增地址数据
            result_table: 结果表名
        """
        # 复用现有的聚类处理逻辑
        address_ids = list(address_data.keys())

        # 查找候选对
        text_candidates = self.similarity_calculator.find_similar_candidates_with_lsh(
            address_data,
            threshold=Config.ALGORITHM_CONFIG.get('text_similarity', {}).get('lsh_threshold', 0.3)
        )

        admin_candidates = self._find_admin_candidates(address_data)
        spatial_candidates = self._find_spatial_candidates(address_data)

        all_candidates = list(set(text_candidates) | set(admin_candidates) | set(spatial_candidates))

        # 计算相似度
        candidate_pairs_with_info = [
            (id1, id2, {'source': 'candidate'})
            for id1, id2 in all_candidates
        ]

        similarity_results = self.similarity_calculator.batch_calculate_similarities(
            candidate_pairs_with_info,
            address_data,
            n_jobs=Config.ALGORITHM_CONFIG.get('performance', {}).get('n_jobs', 8)
        )

        threshold = Config.ALGORITHM_CONFIG.get('clustering', {}).get('similarity_threshold', 0.7)
        similar_pairs = [r for r in similarity_results
                         if r['comprehensive_similarity'] >= threshold]

        # 执行聚类
        similarity_matrix = self.similarity_calculator.create_similarity_matrix(
            address_ids, address_data, similar_pairs
        )

        coordinates = []
        for addr_id in address_ids:
            addr_info = address_data[addr_id]
            if (addr_info.get('latitude') is not None and
                    addr_info.get('longitude') is not None):
                coordinates.append([addr_info['latitude'], addr_info['longitude']])
            else:
                coordinates.append(None)

        coordinates_array = np.array([c if c is not None else [np.nan, np.nan]
                                      for c in coordinates])

        if Config.ALGORITHM_CONFIG.get('clustering', {}).get('use_spatial_constraints', True):
            labels = self.clustering.cluster_with_spatial_constraints(
                similarity_matrix,
                coordinates_array,
                similarity_threshold=threshold,
                spatial_threshold_km=Config.ALGORITHM_CONFIG.get('clustering', {}).get('max_cluster_radius_km', 5)
            )
        else:
            labels = self.clustering.cluster_by_connected_components(
                similarity_matrix,
                threshold=threshold
            )

        # 合并小聚类和验证
        labels = self.clustering.merge_small_clusters(
            labels,
            min_size=Config.ALGORITHM_CONFIG.get('clustering', {}).get('min_cluster_size', 2)
        )

        labels = self.clustering.validate_clusters(labels, address_data, similarity_matrix)

        # 保存结果到数据库
        self._save_new_clustering_results(address_ids, labels, address_data, result_table)

    def _match_new_addresses_to_existing_clusters(self, new_address_data: Dict[int, Dict[str, Any]],
                                                  existing_clusters_df: pd.DataFrame) -> List[Dict]:
        """
        将新增地址匹配到现有聚类

        Args:
            new_address_data: 新增地址数据
            existing_clusters_df: 现有聚类数据

        Returns:
            匹配结果列表
        """
        matched_results = []
        threshold = Config.ALGORITHM_CONFIG.get('clustering', {}).get('similarity_threshold', 0.7)

        # 构建聚类代表信息字典
        cluster_representatives = {}
        for _, row in existing_clusters_df.iterrows():
            # 解析坐标
            latitude, longitude = None, None
            if row['coordinates']:
                try:
                    coords_data = row['coordinates']
                    if isinstance(coords_data, str):
                        coords = json.loads(coords_data)
                    else:
                        coords = coords_data

                    if isinstance(coords, list) and len(coords) >= 2:
                        latitude, longitude = float(coords[0]), float(coords[1])
                    elif isinstance(coords, dict) and 'latitude' in coords and 'longitude' in coords:
                        latitude, longitude = float(coords['latitude']), float(coords['longitude'])
                except Exception as e:
                    logger.warning(f"解析坐标失败: {e}")

            cluster_representatives[row['group_id']] = {
                'standardized': row['standardized_address'] or '',
                'latitude': latitude,
                'longitude': longitude,
                'avg_similarity': float(row.get('avg_similarity', 0)),
                'cluster_size': int(row.get('cluster_size', 1))
            }

        # 对每个新增地址，计算与现有聚类的相似度
        for addr_id, addr_info in new_address_data.items():
            best_match_cluster = None
            best_similarity = 0
            best_similarity_details = {}

            # 与现有聚类进行匹配
            for group_id, rep_info in cluster_representatives.items():
                # 计算相似度
                similarity_result = self.similarity_calculator.calculate_comprehensive_similarity(
                    text1=addr_info['standardized'],
                    text2=rep_info['standardized'],
                    lat1=addr_info['latitude'],
                    lon1=addr_info['longitude'],
                    lat2=rep_info['latitude'],
                    lon2=rep_info['longitude'],
                    comp1=addr_info['parsed'],
                    comp2=addr_info['parsed']  # 使用自身的解析结果作为对比
                )

                if (similarity_result['comprehensive_similarity'] > best_similarity and
                        similarity_result['comprehensive_similarity'] >= threshold):
                    best_similarity = similarity_result['comprehensive_similarity']
                    best_match_cluster = group_id
                    best_similarity_details = similarity_result

            # 记录匹配结果
            matched_results.append({
                'address_id': addr_id,
                'group_id': best_match_cluster,
                'similarity': best_similarity,
                'is_new_cluster': best_match_cluster is None,
                'similarity_details': best_similarity_details,
                'address_info': addr_info
            })

        return matched_results

    def _save_incremental_results(self, matched_results: List[Dict],
                                  new_address_data: Dict[int, Dict[str, Any]],
                                  result_table: str):
        """
        保存增量匹配结果到数据库

        Args:
            matched_results: 匹配结果
            new_address_data: 新增地址数据
            result_table: 结果表名
        """
        # 准备插入数据
        insert_data = []
        new_cluster_counter = 0

        # 统计新聚类
        existing_group_ids = set()
        for result in matched_results:
            if not result['is_new_cluster'] and result['group_id']:
                existing_group_ids.add(result['group_id'])

        # 为新聚类分配ID
        timestamp = int(time.time())

        for result in matched_results:
            addr_id = result['address_id']
            addr_info = result.get('address_info', new_address_data.get(addr_id, {}))

            # 确定聚类ID
            if result['is_new_cluster']:
                new_cluster_counter += 1
                group_id = f"new_cluster_{timestamp}_{new_cluster_counter}"
            else:
                group_id = result['group_id']

            # 构造坐标信息
            coordinates = None
            if addr_info.get('latitude') is not None and addr_info.get('longitude') is not None:
                coordinates = json.dumps([
                    float(addr_info['latitude']),
                    float(addr_info['longitude'])
                ])

            # 构造插入数据
            row_data = {
                'group_id': group_id,
                'group_size': 1,
                'original_address': addr_info.get('original', ''),
                'cleaned_address': addr_info.get('cleaned', addr_info.get('original', '')),  # 使用原始地址作为备选
                'corrected_address': addr_info.get('corrected', addr_info.get('standardized', '')),  # 使用标准化地址作为备选
                'standardized_address': addr_info.get('standardized', ''),
                'similarity_text': addr_info.get('standardized', ''),
                'coordinates': coordinates,
                'similarity_score': float(result.get('similarity', 0)),
                'representative_address': addr_info.get('standardized', ''),  # 添加代表地址
                'weights_config': json.dumps(Config.ALGORITHM_CONFIG.get('weights', {}), ensure_ascii=False)
            }
            insert_data.append(row_data)

        # 批量插入新数据
        if insert_data:
            self.db_handler.save_incremental_results(insert_data, result_table)
            logger.info(f"成功保存 {len(insert_data)} 条增量识别结果")

    def _save_new_clustering_results(self, address_ids: List[int], labels: np.ndarray,
                                     address_data: Dict[int, Dict[str, Any]], result_table: str):
        """
        保存新聚类结果到数据库

        Args:
            address_ids: 地址ID列表
            labels: 聚类标签
            address_data: 地址数据
            result_table: 结果表名
        """
        # 计算每组的大小
        from collections import Counter
        label_counts = Counter(labels)

        # 准备更新数据
        update_data = []
        for idx, label in enumerate(labels):
            if label == -1:  # 噪声点
                continue

            addr_id = address_ids[idx]
            addr_info = address_data[addr_id]
            group_id = f"group_{label}"
            group_size = label_counts[label]

            # 构造坐标信息
            coordinates = None
            if addr_info.get('latitude') is not None and addr_info.get('longitude') is not None:
                coordinates = json.dumps([
                    float(addr_info['latitude']),
                    float(addr_info['longitude'])
                ])

            # 构造字典格式
            row_data = {
                'group_id': group_id,
                'group_size': group_size,
                'original_address': addr_info.get('original', ''),
                'cleaned_address': addr_info.get('cleaned', ''),
                'corrected_address': addr_info.get('corrected', ''),
                'standardized_address': addr_info.get('standardized', ''),
                'similarity_text': addr_info.get('standardized', ''),
                'coordinates': coordinates,
                'similarity_score': 1.0,  # 聚类内完全匹配
                'representative_address': addr_info.get('standardized', ''),  # 添加代表地址
                'weights_config': json.dumps(Config.ALGORITHM_CONFIG.get('weights', {}), ensure_ascii=False)
            }
            update_data.append(row_data)

        # 保存到数据库
        if update_data:
            self.db_handler.save_incremental_results(update_data, result_table)
            logger.info(f"成功保存 {len(update_data)} 条新聚类结果")

    def process_addresses_from_db(self, source_table: str, result_table: str,
                                  limit: int = None):
        """
        从数据库处理地址数据

        Args:
            source_table: 源数据表名
            result_table: 结果表名
            limit: 限制处理条数（用于测试）
        """
        logger.info("开始处理数据库中的地址数据...")

        # 1. 读取地址数据
        start_time = time.time()
        address_df = self.db_handler.fetch_address_data(source_table, limit)

        if address_df.empty:
            logger.info("没有找到地址数据")
            return

        logger.info(f"读取数据耗时: {time.time() - start_time:.2f}秒")

        # 2. 标准化地址
        logger.info("开始地址标准化...")
        start_time = time.time()

        # 提取地址列表
        addresses = address_df['original_address'].fillna('').astype(str).tolist()
        logger.info(f"已提取 {len(addresses)} 个地址")

        # 批量标准化
        normalized_results = self.normalizer.batch_normalize(
            addresses,
            n_jobs=Config.ALGORITHM_CONFIG.get('performance', {}).get('n_jobs', 8)
        )

        logger.info(f"地址标准化完成，耗时: {time.time() - start_time:.2f}秒")

        # 3. 创建结果表
        self.db_handler.create_result_table(result_table)

        # 4. 保存标准化结果到数据库
        logger.info("保存标准化结果到数据库...")
        result_data = []
        for i, (idx, row) in enumerate(address_df.iterrows()):
            norm_result = normalized_results[i]
            result_data.append({
                'normalized_result': norm_result,
                'original_id': row['id']
            })

        self.db_handler.save_results(result_data, result_table)

        # 5. 地理编码
        logger.info("执行地理编码...")
        start_time = time.time()

        # 提取标准化地址用于地理编码
        standardized_addresses = [
            result['standardized'] for result in normalized_results
        ]

        # 批量地理编码
        geocoded_results = self.processor.batch_process(
            standardized_addresses,
            self.geocoder.geocode,
            batch_size=1000,
            desc="地理编码"
        )

        logger.info(f"地理编码完成，耗时: {time.time() - start_time:.2f}秒")

        # 6. 准备地址数据用于相似度计算
        address_data = {}
        for i, (idx, row) in enumerate(address_df.iterrows()):
            norm_result = normalized_results[i]
            geo_result = geocoded_results[i]

            # 从 offline_geocode 中提取坐标
            latitude, longitude = None, None
            if 'offline_geocode' in row and row['offline_geocode']:
                latitude, longitude = self.extract_coordinates_from_offline_geocode(row['offline_geocode'])

            # 如果 offline_geocode 中没有有效坐标，则使用 geocoded_results 中的结果
            if latitude is None or longitude is None:
                if geo_result:
                    latitude = geo_result.get('latitude')
                    longitude = geo_result.get('longitude')

            address_data[row['id']] = {
                'id': row['id'],
                'original': norm_result.get('original', ''),
                'standardized': norm_result.get('standardized', ''),
                'latitude': latitude,
                'longitude': longitude,
                'parsed': norm_result.get('parsed', {}),
                'components': norm_result.get('components', {}),
                'geo_info': geo_result or {}
            }

        # 7. 查找相似候选对
        logger.info("查找相似候选对...")
        start_time = time.time()

        # 使用LSH查找文本相似候选对
        text_candidates = self.similarity_calculator.find_similar_candidates_with_lsh(
            address_data,
            threshold=Config.ALGORITHM_CONFIG.get('text_similarity', {}).get('lsh_threshold', 0.3)
        )
        logger.info(f"找到 {len(text_candidates)} 个文本相似候选对")

        # 基于行政区划的候选对
        admin_candidates = self._find_admin_candidates(address_data)
        logger.info(f"找到 {len(admin_candidates)} 个行政区划候选对")

        # 基于地理空间的候选对
        spatial_candidates = self._find_spatial_candidates(address_data)
        logger.info(f"找到 {len(spatial_candidates)} 个空间候选对")

        # 合并所有候选对（去重）
        all_candidates = list(set(text_candidates) | set(admin_candidates) | set(spatial_candidates))
        logger.info(f"候选对总数: {len(all_candidates):,}")

        logger.info(f"查找候选对耗时: {time.time() - start_time:.2f}秒")

        # 8. 计算相似度
        logger.info("计算相似度...")
        start_time = time.time()

        # 准备候选对数据
        candidate_pairs_with_info = [
            (id1, id2, {'source': 'candidate'})
            for id1, id2 in all_candidates
        ]

        # 批量计算相似度
        # similarity_results = self.similarity_calculator.batch_calculate_similarities(
        #     candidate_pairs_with_info,
        #     address_data,
        #     n_jobs=Config.ALGORITHM_CONFIG.get('performance', {}).get('n_jobs', 8)
        # )

        # 对于大数据量，分批处理相似度计算
        if len(candidate_pairs_with_info) > 10000:
            # 分批处理避免内存压力
            batch_size = 5000
            similarity_results = []

            for i in range(0, len(candidate_pairs_with_info), batch_size):
                batch_pairs = candidate_pairs_with_info[i:i + batch_size]

                batch_results = self.similarity_calculator.batch_calculate_similarities(
                    batch_pairs,
                    address_data,
                    n_jobs=Config.ALGORITHM_CONFIG.get('performance', {}).get('n_jobs', 8)
                )
                similarity_results.extend(batch_results)
        else:
            # 小数据量直接处理
            similarity_results = self.similarity_calculator.batch_calculate_similarities(
                candidate_pairs_with_info,
                address_data,
                n_jobs=Config.ALGORITHM_CONFIG.get('performance', {}).get('n_jobs', 8)
            )

        # 过滤达到阈值的相似对
        threshold = Config.ALGORITHM_CONFIG.get('clustering', {}).get('similarity_threshold', 0.7)
        similar_pairs = [r for r in similarity_results
                         if r['comprehensive_similarity'] >= threshold]

        logger.info(f"相似度计算完成，耗时: {time.time() - start_time:.2f}秒")
        logger.info(f"相似对数量: {len(similar_pairs):,}")

        # 9. 聚类分析
        logger.info("执行聚类分析...")
        start_time = time.time()

        # 创建相似度矩阵
        address_ids = list(address_data.keys())
        # 方案1：创建矩阵构建器：
        similarity_matrix = self.similarity_calculator.create_similarity_matrix(
            address_ids, address_data, similar_pairs
        )
        # 方案2：创建矩阵构建器
        # builder = SimilarityMatrixBuilder()
        # # 转换 similar_pairs 为合适的格式
        # similarity_pairs = [(pair['id1'], pair['id2'], pair['comprehensive_similarity'])
        #                     for pair in similar_pairs]
        # similarity_matrix = builder.build_sparse_similarity_matrix(
        #     len(address_ids), similarity_pairs
        # )

        # 准备坐标数据
        coordinates = []
        for addr_id in address_ids:
            addr_info = address_data[addr_id]
            if (addr_info.get('latitude') is not None and
                    addr_info.get('longitude') is not None):
                coordinates.append([addr_info['latitude'], addr_info['longitude']])
            else:
                coordinates.append(None)

        coordinates_array = np.array([c if c is not None else [np.nan, np.nan]
                                      for c in coordinates])

        # 执行聚类
        if Config.ALGORITHM_CONFIG.get('clustering', {}).get('use_spatial_constraints', True):
            # 带空间约束的聚类
            labels = self.clustering.cluster_with_spatial_constraints(
                similarity_matrix,
                coordinates_array,
                similarity_threshold=threshold,
                spatial_threshold_km=Config.ALGORITHM_CONFIG.get('clustering', {}).get('max_cluster_radius_km', 5)
            )
        else:
            # 仅基于相似度的聚类
            labels = self.clustering.cluster_by_connected_components(
                similarity_matrix,
                threshold=threshold
            )

        # 合并小聚类
        labels = self.clustering.merge_small_clusters(
            labels,
            min_size=Config.ALGORITHM_CONFIG.get('clustering', {}).get('min_cluster_size', 2)
        )

        # 验证聚类结果，拆分不合理聚类
        labels = self.clustering.validate_clusters(labels, address_data, similarity_matrix)

        logger.info(f"聚类分析完成，耗时: {time.time() - start_time:.2f}秒")
        logger.info(f"识别出 {len(set(labels)) - (1 if -1 in labels else 0)} 个聚类组")

        # 计算聚类质量
        quality = self.clustering.calculate_cluster_quality(labels, similarity_matrix)

        # 验证聚类时获取具体地址信息进行详细分析
        # addr_info = self.get_address_by_index(problematic_index, address_ids, address_data)
        # logger.debug(f"Address details: {self.get_address_by_index(idx, address_ids, address_data)}")

        logger.info(f"聚类完成:")
        logger.info(f"  - 聚类数量: {quality['n_clusters']}")
        logger.info(f"  - 类内平均相似度: {quality['avg_similarity_within']:.3f}")
        logger.info(f"  - 类间平均相似度: {quality['avg_similarity_between']:.3f}")
        logger.info(f"  - 轮廓系数: {quality['silhouette_score']:.3f}")

        # 10. 更新聚类信息到数据库
        logger.info("更新聚类信息到数据库...")
        # 将labels转换为address_df的格式
        address_df_with_labels = address_df.copy()
        address_df_with_labels['cluster_id'] = labels
        self.db_handler.update_cluster_info(result_table, labels, address_df_with_labels)

        logger.info("地址相似度识别处理完成！")

    def get_address_by_index(self, index: int, address_ids: List[int],
                             address_data: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        根据索引获取地址信息

        Args:
            index: 地址在数组中的索引
            address_ids: 地址ID列表
            address_data: 地址数据字典

        Returns:
            地址信息字典
        """
        addr_id = address_ids[index]
        return address_data[addr_id]

    def _find_admin_candidates(self, address_data: Dict[int, Dict[str, Any]]) -> List[tuple]:
        """基于行政区划查找候选对"""
        candidates = set()

        # 按省份-城市-区县分组
        admin_groups = {}

        for addr_id, addr_info in address_data.items():
            parsed = addr_info.get('parsed', {})
            admin_key = (
                parsed.get('province', ''),
                parsed.get('city', ''),
                parsed.get('district', '')
            )

            if admin_key not in admin_groups:
                admin_groups[admin_key] = []
            admin_groups[admin_key].append(addr_id)

        # 同组内的地址作为候选对
        for group in admin_groups.values():
            if len(group) > 1:
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        candidates.add((min(group[i], group[j]), max(group[i], group[j])))

        return list(candidates)

    def _find_spatial_candidates(self, address_data: Dict[int, Dict[str, Any]]) -> List[tuple]:
        """基于地理空间查找候选对"""
        candidates = set()

        # 提取有坐标的地址
        addresses_with_coords = []
        for addr_id, addr_info in address_data.items():
            lat = addr_info.get('latitude')
            lon = addr_info.get('longitude')
            if lat is not None and lon is not None:
                addresses_with_coords.append((addr_id, lat, lon))

        if len(addresses_with_coords) < 2:
            return []

        logger.info(f"有坐标的地址: {len(addresses_with_coords)}")

        # 使用网格划分空间，加速查找
        grid_size = 0.1  # 约10公里

        # 创建空间网格
        grid = {}
        for addr_id, lat, lon in addresses_with_coords:
            grid_x = int(lon / grid_size)
            grid_y = int(lat / grid_size)
            grid_key = (grid_x, grid_y)

            if grid_key not in grid:
                grid[grid_key] = []
            grid[grid_key].append((addr_id, lat, lon))

        # 在每个网格及其相邻网格中查找候选对
        max_distance_km = Config.ALGORITHM_CONFIG.get('spatial_similarity', {}).get('max_search_radius_km', 50)

        for (grid_x, grid_y), addresses in grid.items():
            # 检查当前网格和相邻网格
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    neighbor_key = (grid_x + dx, grid_y + dy)

                    if neighbor_key in grid:
                        # 比较当前网格和相邻网格中的地址
                        for addr1_id, lat1, lon1 in addresses:
                            for addr2_id, lat2, lon2 in grid[neighbor_key]:
                                if addr1_id < addr2_id:  # 避免重复
                                    # 快速距离估算
                                    distance_approx = self._approx_distance(lat1, lon1, lat2, lon2)

                                    if distance_approx <= max_distance_km * 1.5:  # 宽松阈值
                                        candidates.add((addr1_id, addr2_id))

        return list(candidates)

    def _approx_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """快速估算距离（公里）"""
        from math import radians, cos, sqrt
        # 简单估算，每度纬度约111公里，每度经度约111*cos(lat)公里
        dlat = abs(lat1 - lat2) * 111
        dlon = abs(lon1 - lon2) * 111 * max(0.5, cos(radians((lat1 + lat2) / 2)))
        return sqrt(dlat ** 2 + dlon ** 2)

    def extract_coordinates_from_offline_geocode(self, offline_geocode_str: str):
        """
        从 offline_geocode JSON 字符串中提取坐标
        优先使用 gcj02，没有则使用 wgs84

        Args:
            offline_geocode_str: offline_geocode JSON 字符串

        Returns:
            tuple: (latitude, longitude) 或 (None, None)
        """
        try:

            data = json.loads(offline_geocode_str)

            if 'offline_geocode' in data:
                offline_data = data['offline_geocode']

                # 优先使用 gcj02 坐标
                if 'gcj02' in offline_data:
                    coords = offline_data['gcj02'].split(',')
                    return float(coords[1]), float(coords[0])  # lat, lon

                # 如果没有 gcj02，使用 wgs84
                elif 'wgs84' in offline_data:
                    coords = offline_data['wgs84'].split(',')
                    return float(coords[1]), float(coords[0])  # lat, lon

        except (json.JSONDecodeError, KeyError, ValueError, IndexError):
            pass

        return None, None

    def close(self):
        """关闭系统"""
        self.db_handler.disconnect()


def main():
    """主函数"""
    from datetime import datetime, timedelta
    parser = argparse.ArgumentParser(description='地址相似度识别系统（数据库版）')
    parser.add_argument('--source-table', default='dwd_cnc_order_address_similarity_match',
                        help='源数据表名')
    parser.add_argument('--result-table', default='address_similarity_results',
                        help='结果表名')
    parser.add_argument('--incremental', action='store_true', default=True,
                        help='启用增量处理模式')
    parser.add_argument('--last-processed-time', type=str,
                        help='上次处理时间（格式：YYYY-MM-DD HH:MM:SS）')
    parser.add_argument('--limit', type=int, help='处理数据条数限制（用于测试）')

    args = parser.parse_args()

    args.last_processed_time = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
    # 创建系统实例
    system = AddressSimilaritySystemDB()

    try:
        # 连接数据库
        system.connect_database()

        # 处理地址数据
        if args.incremental:
            print("增量处理模式")
            last_processed_time = None
            if args.last_processed_time:
                from datetime import datetime
                last_processed_time = datetime.strptime(args.last_processed_time, '%Y-%m-%d %H:%M:%S')

            system.process_incremental_addresses(
                source_table=args.source_table,
                result_table=args.result_table,
                last_processed_time=last_processed_time
            )
        else:
            # 全量处理模式
            print("全量处理模式")
            system.process_addresses_from_db(
                source_table=args.source_table,
                result_table=args.result_table,
                limit=args.limit
            )

    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
    finally:
        # 关闭系统
        system.close()


if __name__ == "__main__":
    exit(main())
