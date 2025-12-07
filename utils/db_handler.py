#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ****************************************************************#
# @Time    : 2025/12/6 0:08
# @Author  : JonHe 
# Function : 
# ****************************************************************#
"""
数据库处理模块
支持从MySQL读取地址数据并存储处理结果
"""
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Any, Optional
import json
from config.config import Config
from utils.logger import setup_logging

# 初始化日志记录器
logger = setup_logging('db_handler.py').get_logger()


class DatabaseHandler:
    """数据库处理器"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.engine = None

    def connect(self):
        """建立数据库连接"""
        try:
            # 添加SSL配置选项
            ssl_args = {}
            if self.config.get('ssl_disabled', False):
                ssl_args = {'ssl_disabled': True}
            connection_string = (
                f"mysql+pymysql://{self.config['user']}:{self.config['password']}"
                f"@{self.config['host']}:{self.config['port']}/{self.config['database']}"
                f"?charset={self.config['charset']}&"
                f"ssl_disabled=true"
            )
            self.engine = create_engine(connection_string, connect_args=ssl_args)
            # 测试连接
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info(f"成功连接到数据库: {self.config['host']}:{self.config['port']}/{self.config['database']}")
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise

    def disconnect(self):
        """关闭数据库连接"""
        if self.engine:
            self.engine.dispose()
            logger.info("数据库连接已关闭")

    def fetch_address_data(self, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        从数据库读取地址数据

        Args:
            table_name: 表名
            limit: 限制读取条数（用于测试）

        Returns:
            包含地址数据的DataFrame
        """
        query = f"""
        SELECT 
            id,
            recipient_address_complete as original_address,
            offline_geocode
        FROM {table_name}
        WHERE recipient_address_complete IS NOT NULL 
          AND recipient_address_complete != ''
        """

        if limit:
            query += f" LIMIT {limit}"

        try:
            df = pd.read_sql(query, self.engine)
            logger.info(f"从数据库读取了 {len(df)} 条地址数据")
            return df
        except Exception as e:
            logger.error(f"读取地址数据失败: {e}")
            raise

    def fetch_new_addresses(self, table_name: str, last_processed_time: datetime = None,
                            limit: Optional[int] = None) -> pd.DataFrame:
        """
        获取新增地址数据

        Args:
            table_name: 表名
            last_processed_time: 上次处理时间
            limit: 限制读取条数

        Returns:
            包含新增地址数据的DataFrame
        """
        query = f"""
        SELECT 
            id,
            recipient_address_complete as original_address,
            offline_geocode,
            create_time
        FROM {table_name}
        WHERE recipient_address_complete IS NOT NULL 
          AND recipient_address_complete != ''
        """

        if last_processed_time:
            query += f" AND create_time > '{last_processed_time}'"

        if limit:
            query += f" LIMIT {limit}"

        try:
            df = pd.read_sql(query, self.engine)
            logger.info(f"获取增量数据:\n{query} ")
            logger.info(f"从数据库读取了 {len(df)} 条新增地址数据")
            return df
        except Exception as e:
            logger.error(f"读取新增地址数据失败: {e}")
            raise

    def fetch_existing_clusters(self, table_name: str) -> pd.DataFrame:
        """
        获取现有的聚类结果

        Args:
            table_name: 结果表名

        Returns:
            包含现有聚类结果的DataFrame
        """
        query = f"""
        SELECT 
            id,
            group_id,
            standardized_address,
            coordinates
        FROM {table_name}
        WHERE group_id IS NOT NULL
        """

        try:
            df = pd.read_sql(query, self.engine)
            logger.info(f"从数据库读取了 {len(df)} 条现有聚类数据")
            return df
        except Exception as e:
            logger.error(f"读取现有聚类数据失败: {e}")
            raise

    def fetch_cluster_representatives(self, table_name: str) -> pd.DataFrame:
        """
        获取每个聚类的代表地址信息
        """
        # 获取每个聚类中最中心的地址作为代表
        query = f"""
        SELECT 
            t1.group_id,
            t1.standardized_address,
            t1.coordinates,
            t1.similarity_score,
            t2.avg_similarity,
            t2.cluster_size
        FROM {table_name} t1
        JOIN (
            SELECT 
                group_id,
                standardized_address,
                AVG(similarity_score) as avg_similarity,
                COUNT(*) as cluster_size
            FROM {table_name}
            WHERE group_id IS NOT NULL
            GROUP BY group_id, standardized_address
        ) t2 ON t1.group_id = t2.group_id
        WHERE t1.group_id IS NOT NULL
        """

        try:
            df = pd.read_sql(query, self.engine)
            logger.info(f"从数据库读取了 {len(df)} 个聚类代表")
            return df
        except Exception as e:
            logger.error(f"读取聚类代表数据失败: {e}")
            raise

    def create_result_table(self, table_name: str):
        """创建结果表（更新版）"""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            group_id VARCHAR(50) COMMENT '分组ID',
            group_size INT COMMENT '组内相似地址数量',
            original_address TEXT COMMENT '原始地址',
            cleaned_address TEXT COMMENT '基础清洗后的地址',
            corrected_address TEXT COMMENT '纠错后的地址',
            standardized_address TEXT COMMENT '标准地址',
            similarity_text TEXT COMMENT '处理后进行文本相似度计算的地址',
            coordinates JSON COMMENT '地址坐标信息',
            similarity_score DECIMAL(5,4) COMMENT '相似度得分',
            representative_address TEXT COMMENT '聚类代表地址',
            weights_config JSON COMMENT '相关参数的权重配置',
            processing_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '处理时间',
            INDEX idx_group_id (group_id),
            INDEX idx_processing_time (processing_time)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """

        try:
            with self.engine.connect() as conn:
                conn.execute(text(create_table_sql))
                conn.commit()
            logger.info(f"结果表 {table_name} 创建成功")
        except Exception as e:
            logger.error(f"创建结果表失败: {e}")
            raise

    def save_results_with_group_info(self, results: List[Dict[str, Any]], table_name: str):
        """
        保存带有聚类信息的处理结果到数据库

        Args:
            results: 处理结果列表
            table_name: 结果表名
        """
        if not results:
            logger.info("没有结果需要保存")
            return

        # 批量插入 - 使用字典键名对应
        insert_sql = f"""
        INSERT INTO {table_name} (
            group_id, group_size, original_address, cleaned_address,
            corrected_address, standardized_address, similarity_text, weights_config
        ) VALUES (
            :group_id, :group_size, :original_address, :cleaned_address,
            :corrected_address, :standardized_address, :similarity_text, :weights_config
        )
        """

        try:
            with self.engine.connect() as conn:
                conn.execute(text(insert_sql), results)
                conn.commit()
            logger.info(f"成功保存 {len(results)} 条结果到表 {table_name}")
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
            raise

    def save_incremental_results(self, results: List[Dict[str, Any]], table_name: str):
        """
        保存增量处理结果到数据库

        Args:
            results: 处理结果列表
            table_name: 结果表名
        """
        if not results:
            logger.info("没有结果需要保存")
            return

        insert_sql = f"""
        INSERT INTO {table_name} (
        group_id, group_size, original_address, cleaned_address,
        corrected_address, standardized_address, similarity_text, 
        coordinates, similarity_score, representative_address, weights_config
        ) VALUES (
            :group_id, :group_size, :original_address, :cleaned_address,
            :corrected_address, :standardized_address, :similarity_text,
            :coordinates, :similarity_score, :representative_address, :weights_config
        )
        """

        try:
            with self.engine.connect() as conn:
                conn.execute(text(insert_sql), results)
                conn.commit()
            logger.info(f"成功保存 {len(results)} 条结果到表 {table_name}")
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
            raise

    def save_results(self, results: List[Dict[str, Any]], table_name: str):
        """
        保存处理结果到数据库

        Args:
            results: 处理结果列表
            table_name: 结果表名
        """
        if not results:
            logger.info("没有结果需要保存")
            return

        # 准备插入数据 - 修改为字典列表格式
        insert_data = []
        for result in results:
            # 提取标准化结果
            normalized_result = result.get('normalized_result', {})

            # 提取聚类结果
            cluster_info = result.get('cluster_info', {})

            # 构造字典而不是元组
            row_data = {
                'group_id': cluster_info.get('group_id', ''),
                'group_size': cluster_info.get('group_size', 0),
                'original_address': normalized_result.get('original', ''),
                'cleaned_address': normalized_result.get('cleaned', ''),
                'corrected_address': normalized_result.get('corrected', ''),
                'standardized_address': normalized_result.get('standardized', ''),
                'similarity_text': normalized_result.get('standardized', ''),  # 用于相似度计算的地址
                'weights_config': json.dumps(Config.ALGORITHM_CONFIG.get('weights', {}), ensure_ascii=False)
            }
            insert_data.append(row_data)

        # 批量插入 - 使用字典键名对应
        insert_sql = f"""
        INSERT INTO {table_name} (
            group_id, group_size, original_address, cleaned_address,
            corrected_address, standardized_address, similarity_text, weights_config
        ) VALUES (
            :group_id, :group_size, :original_address, :cleaned_address,
            :corrected_address, :standardized_address, :similarity_text, :weights_config
        )
        """

        try:
            with self.engine.connect() as conn:
                conn.execute(text(insert_sql), insert_data)
                conn.commit()
            logger.info(f"成功保存 {len(insert_data)} 条结果到表 {table_name}")
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
            raise

    def update_cluster_info(self, table_name: str, labels: List[int],
                            address_data: pd.DataFrame):
        """
        更新聚类信息到结果表

        Args:
            table_name: 结果表名
            labels: 聚类标签
            address_data: 地址数据
        """
        try:
            # 计算每组的大小
            from collections import Counter
            label_counts = Counter(labels)

            # 准备更新数据 - 使用字典格式
            update_data = []
            for idx, label in enumerate(labels):
                if label == -1:  # 噪声点
                    continue

                group_id = f"group_{label}"
                group_size = label_counts[label]

                # 构造字典格式
                row_data = {
                    'group_id': group_id,
                    'group_size': group_size,
                    'id': int(address_data.iloc[idx]['id'])  # 确保是整数类型
                }
                update_data.append(row_data)

            # 批量更新
            update_sql = f"""
            UPDATE {table_name} 
            SET group_id = :group_id, group_size = :group_size 
            WHERE id = :id
            """

            with self.engine.connect() as conn:
                conn.execute(text(update_sql), update_data)
                conn.commit()
            logger.info(f"成功更新 {len(update_data)} 条记录的聚类信息")

        except Exception as e:
            logger.error(f"更新聚类信息失败: {e}")
            raise

    def fetch_address_data_by_page(self, table_name: str, limit: int = 20000, offset: int = 0) -> List[Dict]:
        """从数据库加载地址数据 - 支持分页"""
        logger.info(f"从数据库加载地址数据，限制 {limit} 条，偏移 {offset} 条...")
        query = f"""
        SELECT ROW_NUMBER() OVER (ORDER BY recipient_address_complete) AS id,
               recipient_address_complete as original_address,
               offline_geocode
        FROM (
            SELECT DISTINCT 
                recipient_address_complete,
                offline_geocode,
                ROW_NUMBER() OVER (partition BY recipient_address_complete)as rn
            FROM {table_name}
            WHERE high_school_flag <> '高校'
            AND recipient_address_complete IS NOT NULL
            AND LENGTH(recipient_address_complete) >= 5
        ) t 
        WHERE rn=1
        ORDER BY id
        LIMIT :limit OFFSET :offset
        """

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), {'limit': limit, 'offset': offset})
                columns = result.keys()
                data = [dict(zip(columns, row)) for row in result]

            logger.info(f"成功加载 {len(data)} 条地址数据")
            return data

        except SQLAlchemyError as e:
            logger.error(f"数据加载失败: {e}")
            return []
        except Exception as e:
            logger.error(f"未知错误: {e}")
            return []

    def save_cluster_results(self, table_name: str, cluster_results: List[Dict]):
        """保存聚类结果到数据库"""
        logger.info("保存聚类结果到数据库...")

        try:
            with self.engine.begin() as conn:  # 自动提交事务

                # 批量插入新数据
                if cluster_results:
                    insert_stmt = text(f"""
                    INSERT INTO {table_name} 
                    (id, group_id, original_address, processed_address, customer_code, order_code, coordinates, group_size
                    ,text_weight,spatial_weight,spatial_threshold_km,dbscan_eps,dbscan_min_samples)
                    VALUES (:id, :group_id, :original_address, :processed_address, :customer_code, :order_code, :coordinates, :group_size
                    ,:text_weight, :spatial_weight,:spatial_threshold_km,:dbscan_eps, :dbscan_min_samples)
                    """)

                    # 分批次插入以避免内存问题
                    batch_size = 2000
                    total_batches = (len(cluster_results) + batch_size - 1) // batch_size

                    for i in tqdm(range(0, len(cluster_results), batch_size), desc="保存数据"):
                        batch = cluster_results[i:i + batch_size]
                        conn.execute(insert_stmt, batch)

                    logger.info(f"成功保存 {len(cluster_results)} 条聚类结果")
                else:
                    logger.info("没有聚类结果需要保存")

        except SQLAlchemyError as e:
            logger.error(f"保存聚类结果失败: {e}")
            raise
        except Exception as e:
            logger.error(f"未知错误: {e}")
            raise

    def update_original_table_with_group_info(self, table_name: str, cluster_results: List[Dict]):
        """将group_id和group_size回刷到原始表"""
        logger.info("开始回刷group_id和group_size到原始表...")

        try:
            with self.engine.begin() as conn:
                # 分批更新以避免内存问题
                batch_size = 2000

                for i in tqdm(range(0, len(cluster_results), batch_size), desc="回刷数据"):
                    batch = cluster_results[i:i + batch_size]

                    # 构建批量更新语句
                    update_cases_group_id = []
                    update_cases_group_size = []
                    address_list = []

                    for result in batch:
                        original_address = result['original_address']
                        group_id = result['group_id']
                        group_size = result['group_size']

                        update_cases_group_id.append(f"WHEN '{original_address}' THEN {group_id}")
                        update_cases_group_size.append(f"WHEN '{original_address}' THEN {group_size}")
                        address_list.append(original_address)

                    if not address_list:
                        continue

                    # 构造SQL更新语句
                    address_conditions = "', '".join(address_list)
                    group_id_cases = " ".join(update_cases_group_id)
                    group_size_cases = " ".join(update_cases_group_size)

                    update_query = f"""
                    UPDATE {table_name}
                    SET group_id = CASE recipient_address_complete 
                                    {group_id_cases} 
                                  END,
                        group_size = CASE recipient_address_complete 
                                      {group_size_cases} 
                                    END
                    WHERE recipient_address_complete IN ('{address_conditions}')
                    """

                    conn.execute(text(update_query))

                logger.info(f"成功回刷 {len(cluster_results)} 条记录的group_id和group_size到原始表")

        except SQLAlchemyError as e:
            logger.error(f"回刷数据到原始表失败: {e}")
            raise
        except Exception as e:
            logger.error(f"回刷数据时发生未知错误: {e}")
            raise
