#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2025/12/5 15:13
# @Author  : hejun
"""
主程序入口：百万级地址相似度识别与聚类系统
"""
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

import argparse
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from config.config import Config
from core.address_normalizer import AddressStandardizationPipeline
from core.geocoding_integration import GeocodingCHNIntegration, MultiSourceGeocoder
from core.similarity_calculator import MultiDimensionalSimilarityCalculator
from core.clustering import AddressClustering
from utils.parallel_processor import ParallelProcessor, MemoryOptimizedProcessor
from utils.visualization import VisualizationTools
from utils.logger import setup_logging

# 初始化日志记录器
logger = setup_logging('main-bak.py').get_logger()


class AddressSimilaritySystem:
    """地址相似度系统（主类）"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化地址相似度系统

        Args:
            config: 配置字典，None则使用默认配置
        """
        self.config = config or Config.ALGORITHM_CONFIG

        # 初始化各模块
        self.parallel_processor = ParallelProcessor(
            n_jobs=self.config['performance']['n_jobs'],
            max_memory_gb=32
        )

        self.memory_optimizer = MemoryOptimizedProcessor(
            max_memory_gb=32
        )

        self.address_normalizer = AddressStandardizationPipeline(
            self.config['normalization']
        )

        # 地理编码器
        if self.config['geocoding']['use_geocoding_chn']:
            geocoding_path = self.config['geocoding'].get('geocoding_chn_path')
            self.geocoder = GeocodingCHNIntegration(data_path=geocoding_path)
        else:
            self.geocoder = MultiSourceGeocoder()

        # 相似度计算器
        self.similarity_calculator = MultiDimensionalSimilarityCalculator(
            self.config
        )

        # 聚类器
        self.clustering = AddressClustering(self.config)

        # 可视化工具
        self.visualization = VisualizationTools()

        # 状态跟踪
        self.processing_stats = {
            'start_time': None,
            'end_time': None,
            'total_addresses': 0,
            'normalized_addresses': 0,
            'geocoded_addresses': 0,
            'candidate_pairs': 0,
            'similar_pairs': 0,
            'clusters_found': 0,
            'memory_peak_mb': 0
        }

    def load_data(self, input_file: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        加载数据

        Args:
            input_file: 输入文件路径
            sample_size: 采样大小（用于测试）

        Returns:
            加载的DataFrame
        """
        logger.info(f"加载数据: {input_file}")

        # 支持多种格式
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file, encoding='utf-8')
        elif input_file.endswith('.parquet'):
            df = pd.read_parquet(input_file)
        elif input_file.endswith('.xlsx') or input_file.endswith('.xls'):
            df = pd.read_excel(input_file)
        else:
            raise ValueError(f"不支持的文件格式: {input_file}")

        # 采样（用于测试）
        if sample_size and sample_size < len(df):
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
            logger.info(f"采样 {len(df)} 条数据用于测试")

        self.processing_stats['total_addresses'] = len(df)

        return df

    def preprocess_addresses(self, df: pd.DataFrame,
                             address_column: str = 'address',
                             lon_column: Optional[str] = None,
                             lat_column: Optional[str] = None) -> pd.DataFrame:
        """
        预处理地址数据

        Args:
            df: 输入DataFrame
            address_column: 地址列名
            lon_column: 经度列名
            lat_column: 纬度列名

        Returns:
            预处理后的DataFrame
        """
        logger.info("\n=== 地址预处理 ===")

        # 1. 地址标准化
        logger.info("1. 地址标准化...")
        df = self.address_normalizer.process_dataframe(df, address_column)
        self.processing_stats['normalized_addresses'] = df['success'].sum()

        # 2. 地理编码（如果提供了经纬度列，则验证；否则进行地理编码）
        logger.info("2. 地理编码...")

        if lon_column and lat_column and lon_column in df.columns and lat_column in df.columns:
            # 已有经纬度，验证和补全
            df['longitude'] = df[lon_column].astype(float)
            df['latitude'] = df[lat_column].astype(float)

            # 逆地理编码获取结构化信息
            logger.info("  逆地理编码获取地址信息...")
            geocoded_results = self.parallel_processor.batch_process(
                list(zip(df['latitude'], df['longitude'])),
                lambda x: self.geocoder.reverse_geocode(x[0], x[1]),
                batch_size=5000,
                desc="逆地理编码"
            )

            df['geo_info'] = geocoded_results

        else:
            # 没有经纬度，进行地理编码
            logger.info("  正向地理编码...")
            geocoded_results = self.parallel_processor.batch_process(
                df['original'].tolist(),
                self.geocoder.geocode,
                batch_size=5000,
                desc="地理编码"
            )

            # 提取经纬度
            df['longitude'] = [r.get('longitude') if r else None for r in geocoded_results]
            df['latitude'] = [r.get('latitude') if r else None for r in geocoded_results]
            df['geo_info'] = geocoded_results

        self.processing_stats['geocoded_addresses'] = df['longitude'].notna().sum()

        # 3. 添加ID
        df['id'] = range(len(df))

        logger.info(f"预处理完成: {len(df)} 条地址")
        logger.info(f"  - 标准化成功: {self.processing_stats['normalized_addresses']}")
        logger.info(f"  - 地理编码成功: {self.processing_stats['geocoded_addresses']}")

        return df

    def find_similar_candidates(self, df: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        查找相似候选对

        Args:
            df: 预处理后的DataFrame

        Returns:
            候选对列表
        """
        logger.info("\n=== 查找相似候选对 ===")

        # 将数据转换为字典格式
        address_data = {}
        for _, row in df.iterrows():
            address_data[row['id']] = {
                'id': row['id'],
                'original': row['original'],
                'standardized': row['standardized'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'parsed': row.get('parsed', {}),
                'components': row.get('components', {}),
                'geo_info': row.get('geo_info', {})
            }

        # 方法1: 使用LSH查找文本相似的候选对
        logger.info("1. 使用LSH查找文本相似候选对...")
        text_candidates = self.similarity_calculator.find_similar_candidates_with_lsh(
            address_data,
            threshold=self.config['text_similarity']['lsh_threshold']
        )
        logger.info(f"   找到 {len(text_candidates)} 个文本相似候选对")

        # 方法2: 基于行政区划的候选对
        logger.info("2. 基于行政区划查找候选对...")
        admin_candidates = self._find_admin_candidates(address_data)
        logger.info(f"   找到 {len(admin_candidates)} 个行政区划候选对")

        # 方法3: 基于地理空间的候选对
        logger.info("3. 基于地理空间查找候选对...")
        spatial_candidates = self._find_spatial_candidates(address_data)
        logger.info(f"   找到 {len(spatial_candidates)} 个空间候选对")

        # 合并所有候选对（去重）
        all_candidates = set(text_candidates) | set(admin_candidates) | set(spatial_candidates)
        self.processing_stats['candidate_pairs'] = len(all_candidates)

        logger.info(f"候选对总数: {len(all_candidates):,}")

        return list(all_candidates)

    def _find_admin_candidates(self, address_data: Dict[int, Dict[str, Any]]) -> List[Tuple[int, int]]:
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

    def _find_spatial_candidates(self, address_data: Dict[int, Dict[str, Any]]) -> List[Tuple[int, int]]:
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

        logger.info(f"   有坐标的地址: {len(addresses_with_coords)}")

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
        max_distance_km = self.config['spatial_similarity']['max_search_radius_km']

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
        # 简单估算，每度纬度约111公里，每度经度约111*cos(lat)公里
        dlat = abs(lat1 - lat2) * 111
        dlon = abs(lon1 - lon2) * 111 * max(0.5, np.cos(np.radians((lat1 + lat2) / 2)))
        return np.sqrt(dlat ** 2 + dlon ** 2)

    def calculate_similarities(self, df: pd.DataFrame,
                               candidate_pairs: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
        """
        计算相似度

        Args:
            df: 预处理后的DataFrame
            candidate_pairs: 候选对列表

        Returns:
            相似度结果列表
        """
        logger.info("\n=== 计算相似度 ===")

        # 准备地址数据
        address_data = {}
        for _, row in df.iterrows():
            address_data[row['id']] = {
                'id': row['id'],
                'original': row['original'],
                'standardized': row['standardized'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'parsed': row.get('parsed', {}),
                'components': row.get('components', {}),
                'geo_info': row.get('geo_info', {})
            }

        # 准备候选对数据
        candidate_pairs_with_info = [
            (id1, id2, {'source': 'candidate'})
            for id1, id2 in candidate_pairs
        ]

        # 批量计算相似度
        logger.info(f"计算 {len(candidate_pairs):,} 个候选对的相似度...")

        similarity_results = self.similarity_calculator.batch_calculate_similarities(
            candidate_pairs_with_info,
            address_data,
            n_jobs=self.config['performance']['n_jobs']
        )

        # 过滤达到阈值的相似对
        threshold = self.config['clustering']['similarity_threshold']
        similar_pairs = [r for r in similarity_results
                         if r['comprehensive_similarity'] >= threshold]

        self.processing_stats['similar_pairs'] = len(similar_pairs)

        logger.info(f"相似度计算完成:")
        logger.info(f"  - 候选对总数: {len(candidate_pairs):,}")
        logger.info(f"  - 相似对数量: {len(similar_pairs):,}")
        logger.info(f"  - 相似度阈值: {threshold}")

        return similar_pairs

    def cluster_addresses(self, df: pd.DataFrame,
                          similarity_pairs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        地址聚类

        Args:
            df: 预处理后的DataFrame
            similarity_pairs: 相似度对列表

        Returns:
            聚类后的DataFrame
        """
        logger.info("\n=== 地址聚类 ===")

        # 创建相似度矩阵
        logger.info("1. 创建相似度矩阵...")
        address_ids = df['id'].tolist()

        address_data_list = []
        for _, row in df.iterrows():
            address_data_list.append({
                'id': row['id'],
                'original': row['original'],
                'standardized': row['standardized'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'parsed': row.get('parsed', {}),
                'components': row.get('components', {})
            })

        # 转换为字典格式
        address_data_dict = {item['id']: item for item in address_data_list}

        similarity_matrix = self.similarity_calculator.create_similarity_matrix(
            address_ids, address_data_dict, similarity_pairs
        )

        # 准备坐标数据
        coordinates = []
        for addr_id in address_ids:
            addr_info = address_data_dict[addr_id]
            if (addr_info.get('latitude') is not None and
                    addr_info.get('longitude') is not None):
                coordinates.append([addr_info['latitude'], addr_info['longitude']])
            else:
                coordinates.append(None)

        coordinates_array = np.array([c if c is not None else [np.nan, np.nan]
                                      for c in coordinates])

        # 聚类
        logger.info("2. 执行聚类...")

        if self.config['clustering'].get('use_spatial_constraints', True):
            # 带空间约束的聚类
            labels = self.clustering.cluster_with_spatial_constraints(
                similarity_matrix,
                coordinates_array,
                similarity_threshold=self.config['clustering']['similarity_threshold'],
                spatial_threshold_km=self.config['clustering']['max_cluster_radius_km']
            )
        else:
            # 仅基于相似度的聚类
            labels = self.clustering.cluster_by_connected_components(
                similarity_matrix,
                threshold=self.config['clustering']['similarity_threshold']
            )

        # 合并小聚类
        logger.info("3. 优化聚类结果...")
        labels = self.clustering.merge_small_clusters(
            labels,
            min_size=self.config['clustering']['min_cluster_size']
        )

        # 计算聚类质量
        quality = self.clustering.calculate_cluster_quality(labels, similarity_matrix)

        logger.info(f"聚类完成:")
        logger.info(f"  - 聚类数量: {quality['n_clusters']}")
        logger.info(f"  - 类内平均相似度: {quality['avg_similarity_within']:.3f}")
        logger.info(f"  - 类间平均相似度: {quality['avg_similarity_between']:.3f}")
        logger.info(f"  - 轮廓系数: {quality['silhouette_score']:.3f}")

        # 添加聚类标签到DataFrame
        df['cluster_id'] = labels

        self.processing_stats['clusters_found'] = quality['n_clusters']

        return df, quality

    def generate_results(self, df: pd.DataFrame,
                         similarity_pairs: List[Dict[str, Any]],
                         output_prefix: str):
        """
        生成结果文件

        Args:
            df: 聚类后的DataFrame
            similarity_pairs: 相似度对列表
            output_prefix: 输出文件前缀
        """
        logger.info("\n=== 生成结果文件 ===")

        output_dir = Config.OUTPUT_DIR
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. 保存聚类结果
        cluster_file = output_dir / f"{output_prefix}_clusters_{timestamp}.csv"
        df.to_csv(cluster_file, index=False, encoding='utf-8-sig')
        logger.info(f"1. 聚类结果保存到: {cluster_file}")

        # 2. 保存聚类摘要
        cluster_summary = self.clustering.create_cluster_summary(
            df['cluster_id'].values,
            df.to_dict('records')
        )

        summary_file = output_dir / f"{output_prefix}_cluster_summary_{timestamp}.csv"
        cluster_summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
        logger.info(f"2. 聚类摘要保存到: {summary_file}")

        # 3. 保存相似度对
        similarity_df = pd.DataFrame(similarity_pairs)
        similarity_file = output_dir / f"{output_prefix}_similarities_{timestamp}.csv"
        similarity_df.to_csv(similarity_file, index=False, encoding='utf-8-sig')
        logger.info(f"3. 相似度对保存到: {similarity_file}")

        # 4. 保存处理统计
        self.processing_stats['end_time'] = time.time()
        duration = self.processing_stats['end_time'] - self.processing_stats['start_time']

        stats = {
            'processing_stats': self.processing_stats,
            'duration_seconds': duration,
            'addresses_per_second': self.processing_stats['total_addresses'] / duration,
            'config': self.config,
            'quality_metrics': self.clustering.calculate_cluster_quality(
                df['cluster_id'].values,
                self.similarity_calculator.create_similarity_matrix(
                    df['id'].tolist(),
                    {row['id']: row for _, row in df.iterrows()},
                    similarity_pairs
                )
            ),
            'geocoding_stats': self.geocoder.get_stats() if hasattr(self.geocoder, 'get_stats') else {}
        }

        stats_file = output_dir / f"{output_prefix}_processing_stats_{timestamp}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"4. 处理统计保存到: {stats_file}")

        # 5. 生成可视化报告
        try:
            vis_file = output_dir / f"{output_prefix}_visualization_{timestamp}.html"
            self.visualization.create_interactive_report(
                df, similarity_df, str(vis_file)
            )
            logger.info(f"5. 可视化报告保存到: {vis_file}")
        except Exception as e:
            logger.error(f"生成可视化报告失败: {e}")

        logger.infologger.info(f"\n✅ 所有结果已保存到: {output_dir}")

    def run_pipeline(self, input_file: str,
                     output_prefix: str = "address_similarity",
                     address_column: str = "address",
                     lon_column: Optional[str] = None,
                     lat_column: Optional[str] = None,
                     sample_size: Optional[int] = None):
        """
        运行完整处理管道

        Args:
            input_file: 输入文件路径
            output_prefix: 输出文件前缀
            address_column: 地址列名
            lon_column: 经度列名
            lat_column: 纬度列名
            sample_size: 采样大小（用于测试）
        """
        logger.info("=" * 60)
        logger.info("百万级地址相似度识别与聚类系统")
        logger.info(f"开始时间: {datetime.now()}")
        logger.info("=" * 60)

        self.processing_stats['start_time'] = time.time()

        try:
            # 1. 加载数据
            df = self.load_data(input_file, sample_size)

            # 2. 预处理
            df = self.preprocess_addresses(df, address_column, lon_column, lat_column)

            # 3. 查找相似候选对
            candidate_pairs = self.find_similar_candidates(df)

            # 4. 计算相似度
            similarity_pairs = self.calculate_similarities(df, candidate_pairs)

            # 5. 聚类
            df, quality = self.cluster_addresses(df, similarity_pairs)

            # 6. 生成结果
            self.generate_results(df, similarity_pairs, output_prefix)

            # 7. 打印摘要
            self.print_summary(df, quality)

        except Exception as e:
            logger.error(f"处理失败: {e}")
            import traceback
            traceback.print_exc()
            raiselogger.info

    def print_summary(self, df: pd.DataFrame, quality: Dict[str, Any]):
        """打印处理摘要"""
        duration = time.time() - self.processing_stats['start_time']

        logger.info("\n" + "=" * 60)
        logger.info("处理完成!")
        logger.info("=" * 60)
        logger.info(f"📊 统计摘要:")
        logger.info(f"   总地址数: {self.processing_stats['total_addresses']:,}")
        logger.info(f"   标准化成功: {self.processing_stats['normalized_addresses']:,}")
        logger.info(f"   地理编码成功: {self.processing_stats['geocoded_addresses']:,}")
        logger.info(f"   候选对数量: {self.processing_stats['candidate_pairs']:,}")
        logger.info(f"   相似对数量: {self.processing_stats['similar_pairs']:,}")
        logger.info(f"   聚类数量: {self.processing_stats['clusters_found']:,}")
        logger.info(f"   处理时间: {duration:.2f}秒")
        logger.info(f"   处理速度: {self.processing_stats['total_addresses'] / duration:.1f} 地址/秒")
        logger.info(f"\n🎯 聚类质量:")
        logger.info(f"   轮廓系数: {quality.get('silhouette_score', 0):.3f}")
        logger.info(f"   类内平均相似度: {quality.get('avg_similarity_within', 0):.3f}")
        logger.info(f"   类间平均相似度: {quality.get('avg_similarity_between', 0):.3f}")
        logger.info(f"   聚类大小分布:")

        # 分析聚类大小分布
        cluster_sizes = df['cluster_id'].value_counts()
        size_stats = cluster_sizes.describe()

        for stat, value in size_stats.items():
            logger.info(f"     {stat}: {value:.1f}")

        logger.info(f"\n📁 输出文件保存在: {Config.OUTPUT_DIR}")
        logger.infologger.info("=" * 60)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='百万级地址相似度识别与聚类系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
      使用示例:
      # 测试运行（1万条数据）
      python main-bak.py --input data/sample.csv --output test_run --sample 10000
    
      # 全量运行
      python main-bak.py --input data/addresses.csv --output full_run
    
      # 指定列名
      python main-bak.py --input data/addresses.csv --address-col address --lon-col lng --lat-col lat
        """
    )

    parser.add_argument('--input', required=True,
                        help='输入文件路径（CSV格式）')
    parser.add_argument('--output', default='result',
                        help='输出文件前缀')
    parser.add_argument('--address-col', default='address',
                        help='地址列名')
    parser.add_argument('--lon-col',
                        help='经度列名（可选）')
    parser.add_argument('--lat-col',
                        help='纬度列名（可选）')
    parser.add_argument('--sample', type=int,
                        help='采样大小（用于测试）')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='相似度阈值（默认: 0.7）')
    parser.add_argument('--jobs', type=int, default=28,
                        help='并行任务数（默认: 28）')
    parser.add_argument('--visualize', action='store_true',
                        help='生成可视化报告')
    parser.add_argument('--config',
                        help='配置文件路径（JSON格式）')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()

    logger.info("🚀 启动地址相似度识别与聚类系统")
    logger.info(f"输入文件: {args.input}")
    logger.info(f"输出前缀: {args.output}")
    logger.info(f"并行任务数: {args.jobs}")
    logger.info(f"相似度阈值: {args.threshold}")

    # 加载配置
    config = Config.ALGORITHM_CONFIG.copy()

    # 更新命令行参数
    config['clustering']['similarity_threshold'] = args.threshold
    config['performance']['n_jobs'] = args.jobs

    # 如果有配置文件，加载并覆盖
    if args.config:
        import json
        with open(args.config, 'r', encoding='utf-8') as f:
            user_config = json.load(f)
        config.update(user_config)

    # 创建系统实例
    system = AddressSimilaritySystem(config)

    # 运行处理管道
    try:
        system.run_pipeline(
            input_file=args.input,
            output_prefix=args.output,
            address_column=args.address_col,
            lon_column=args.lon_col,
            lat_column=args.lat_col,
            sample_size=args.sample
        )

        # 如果需要可视化
        if args.visualize:
            logger.info("\n📊 生成可视化报告...")
            from utils.visualization import VisualizationTools
            viz = VisualizationTools()

            # 加载结果数据
            output_dir = Config.OUTPUT_DIR
            results_files = list(output_dir.glob(f"{args.output}_clusters_*.csv"))

            if results_files:
                latest_file = max(results_files, key=lambda x: x.stat().st_mtime)
                df = pd.read_csv(latest_file)

                # 生成可视化
                viz.create_interactive_report(df, output_prefix=args.output)
                logger.info("✅ 可视化报告已生成")

    except KeyboardInterrupt:
        logger.error("\n⏹️ 用户中断处理")
    except Exception as e:
        logger.error(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    exit(main())
