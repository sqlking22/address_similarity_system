#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2025/12/5 15:27
# @Author  : hejun
"""
可视化模块
生成地址聚类的可视化报告
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from folium.plugins import MarkerCluster, HeatMap
import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class VisualizationTools:
    """可视化工具类"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def plot_cluster_distribution(self, df: pd.DataFrame, output_file: str = None):
        """绘制聚类分布图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 聚类大小分布
        cluster_sizes = df['cluster_id'].value_counts()
        axes[0, 0].hist(cluster_sizes.values, bins=50, log=True, alpha=0.7)
        axes[0, 0].set_title('Cluster Size Distribution (Log Scale)')
        axes[0, 0].set_xlabel('Cluster Size')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 聚类大小箱线图
        axes[0, 1].boxplot(cluster_sizes.values, vert=False)
        axes[0, 1].set_title('Cluster Size Box Plot')
        axes[0, 1].set_xlabel('Size')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 地址长度分布
        df['address_length'] = df['original'].fillna('').astype(str).str.len()
        axes[1, 0].hist(df['address_length'], bins=50, alpha=0.7)
        axes[1, 0].set_title('Address Length Distribution')
        axes[1, 0].set_xlabel('Length (characters)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 聚类数量随时间变化（如果有时间戳）
        if 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            cluster_count_by_date = df.groupby('date')['cluster_id'].nunique()
            axes[1, 1].plot(cluster_count_by_date.index, cluster_count_by_date.values,
                            marker='o', linewidth=2)
            axes[1, 1].set_title('Number of Clusters Over Time')
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('Number of Clusters')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            # 显示标准化成功率
            if 'success' in df.columns:
                success_rate = df['success'].mean() * 100
                axes[1, 1].pie([success_rate, 100 - success_rate],
                               labels=['Success', 'Failure'],
                               autopct='%1.1f%%',
                               colors=['#4CAF50', '#F44336'])
                axes[1, 1].set_title(f'Normalization Success Rate: {success_rate:.1f}%')

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"聚类分布图已保存: {output_file}")

        plt.show()

        return fig

    def plot_geographic_distribution(self, df: pd.DataFrame, output_file: str = None):
        """绘制地理分布图"""
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            print("警告: 数据中没有经纬度信息，无法绘制地理分布图")
            return None

        # 筛选有效坐标
        valid_coords = df.dropna(subset=['latitude', 'longitude'])

        if len(valid_coords) == 0:
            print("警告: 没有有效的坐标数据")
            return None

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # 1. 散点图
        scatter = axes[0].scatter(
            valid_coords['longitude'],
            valid_coords['latitude'],
            c=valid_coords['cluster_id'] if 'cluster_id' in valid_coords.columns else 'blue',
            cmap='tab20',
            alpha=0.6,
            s=20,
            edgecolors='w',
            linewidth=0.5
        )
        axes[0].set_title('Geographic Distribution of Addresses')
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')

        if 'cluster_id' in valid_coords.columns:
            plt.colorbar(scatter, ax=axes[0], label='Cluster ID')

        # 2. 密度图
        if len(valid_coords) > 1000:
            # 对于大数据集，使用hexbin
            hb = axes[1].hexbin(
                valid_coords['longitude'],
                valid_coords['latitude'],
                gridsize=50,
                cmap='YlOrRd',
                mincnt=1
            )
            cb = plt.colorbar(hb, ax=axes[1])
            cb.set_label('Count')
        else:
            # 对于小数据集，使用kde
            try:
                sns.kdeplot(
                    x=valid_coords['longitude'],
                    y=valid_coords['latitude'],
                    ax=axes[1],
                    fill=True,
                    cmap='YlOrRd',
                    alpha=0.7
                )
            except:
                # 如果kde失败，使用散点图
                axes[1].scatter(
                    valid_coords['longitude'],
                    valid_coords['latitude'],
                    alpha=0.1,
                    s=5
                )

        axes[1].set_title('Address Density Map')
        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"地理分布图已保存: {output_file}")

        plt.show()

        return fig

    def plot_similarity_distribution(self, similarity_df: pd.DataFrame, output_file: str = None):
        """绘制相似度分布图"""
        if similarity_df.empty or 'comprehensive_similarity' not in similarity_df.columns:
            print("警告: 没有相似度数据")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 综合相似度分布
        axes[0, 0].hist(similarity_df['comprehensive_similarity'],
                        bins=50, alpha=0.7, color='skyblue')
        axes[0, 0].axvline(x=0.7, color='red', linestyle='--', alpha=0.7,
                           label='Threshold (0.7)')
        axes[0, 0].set_title('Comprehensive Similarity Distribution')
        axes[0, 0].set_xlabel('Similarity')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 文本相似度分布
        if 'text_similarity' in similarity_df.columns:
            axes[0, 1].hist(similarity_df['text_similarity'],
                            bins=50, alpha=0.7, color='lightgreen')
            axes[0, 1].set_title('Text Similarity Distribution')
            axes[0, 1].set_xlabel('Similarity')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)

        # 3. 空间相似度分布
        if 'spatial_similarity' in similarity_df.columns:
            axes[1, 0].hist(similarity_df['spatial_similarity'],
                            bins=50, alpha=0.7, color='lightcoral')
            axes[1, 0].set_title('Spatial Similarity Distribution')
            axes[1, 0].set_xlabel('Similarity')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)

        # 4. 距离分布
        if 'distance_km' in similarity_df.columns:
            valid_distances = similarity_df['distance_km'].dropna()
            if len(valid_distances) > 0:
                axes[1, 1].hist(valid_distances,
                                bins=50, alpha=0.7, color='gold')
                axes[1, 1].axvline(x=10, color='red', linestyle='--', alpha=0.7,
                                   label='10km Threshold')
                axes[1, 1].set_title('Distance Distribution')
                axes[1, 1].set_xlabel('Distance (km)')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"相似度分布图已保存: {output_file}")

        plt.show()

        return fig

    def create_interactive_map(self, df: pd.DataFrame, output_file: str = None):
        """创建交互式地图"""
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            print("警告: 没有坐标数据，无法创建地图")
            return None

        valid_coords = df.dropna(subset=['latitude', 'longitude'])

        if len(valid_coords) == 0:
            print("警告: 没有有效的坐标数据")
            return None

        # 计算中心点
        center_lat = valid_coords['latitude'].mean()
        center_lon = valid_coords['longitude'].mean()

        # 创建地图
        m = folium.Map(location=[center_lat, center_lon],
                       zoom_start=10,
                       control_scale=True)

        # 添加标记集群
        marker_cluster = MarkerCluster().add_to(m)

        # 为不同聚类分配颜色
        if 'cluster_id' in valid_coords.columns:
            unique_clusters = valid_coords['cluster_id'].unique()
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
            color_map = {cluster: f'#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}'
                         for cluster, (r, g, b, _) in zip(unique_clusters, colors)}
        else:
            color_map = {None: '#3388ff'}

        # 添加标记
        for _, row in valid_coords.iterrows():
            cluster_id = row.get('cluster_id')
            color = color_map.get(cluster_id, '#3388ff')

            # 弹出框内容
            popup_text = f"""
            <div style="font-family: Arial; min-width: 200px;">
                <b>地址:</b> {row.get('original', '')[:50]}...<br>
                <b>标准化:</b> {row.get('standardized', '')[:30]}...<br>
                """

            if cluster_id is not None:
                popup_text += f"<b>聚类ID:</b> {cluster_id}<br>"

            if 'components' in row and isinstance(row['components'], dict):
                comp = row['components']
                if comp.get('road'):
                    popup_text += f"<b>道路:</b> {comp['road']}<br>"
                if comp.get('building'):
                    popup_text += f"<b>建筑:</b> {comp['building']}<br>"

            popup_text += f"""
                <b>坐标:</b> {row['latitude']:.4f}, {row['longitude']:.4f}<br>
            </div>
            """

            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                popup=folium.Popup(popup_text, max_width=300),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                weight=1
            ).add_to(marker_cluster)

        # 添加热力图图层
        heat_data = [[row['latitude'], row['longitude']]
                     for _, row in valid_coords.iterrows()]
        HeatMap(heat_data, radius=10, blur=15, max_zoom=1).add_to(m)

        # 添加图层控制
        folium.LayerControl().add_to(m)

        if output_file:
            m.save(output_file)
            print(f"交互式地图已保存: {output_file}")

        return m

    def create_interactive_report(self, df: pd.DataFrame,
                                  similarity_df: pd.DataFrame = None,
                                  output_prefix: str = 'report'):
        """创建交互式HTML报告"""
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        # 创建子图布局
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('地理分布', '聚类大小分布',
                            '相似度分布', '地址长度分布',
                            '聚类质量指标', '处理统计'),
            specs=[[{'type': 'scattergeo'}, {'type': 'bar'}],
                   [{'type': 'histogram'}, {'type': 'histogram'}],
                   [{'type': 'indicator'}, {'type': 'table'}]]
        )

        # 1. 地理分布图
        if 'latitude' in df.columns and 'longitude' in df.columns:
            valid_coords = df.dropna(subset=['latitude', 'longitude'])
            if len(valid_coords) > 0:
                color_col = 'cluster_id' if 'cluster_id' in df.columns else None

                fig.add_trace(
                    go.Scattergeo(
                        lon=valid_coords['longitude'],
                        lat=valid_coords['latitude'],
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=valid_coords[color_col] if color_col else 'blue',
                            colorscale='Viridis',
                            showscale=bool(color_col),
                            colorbar=dict(title="Cluster ID" if color_col else None)
                        ),
                        text=valid_coords['original'].str[:50],
                        name='地址点'
                    ),
                    row=1, col=1
                )

        # 2. 聚类大小分布
        if 'cluster_id' in df.columns:
            cluster_sizes = df['cluster_id'].value_counts()
            fig.add_trace(
                go.Bar(
                    x=list(range(len(cluster_sizes))),
                    y=cluster_sizes.values,
                    name='聚类大小',
                    marker_color='lightblue'
                ),
                row=1, col=2
            )

        # 3. 相似度分布
        if similarity_df is not None and 'comprehensive_similarity' in similarity_df.columns:
            fig.add_trace(
                go.Histogram(
                    x=similarity_df['comprehensive_similarity'],
                    nbinsx=50,
                    name='综合相似度',
                    marker_color='lightgreen'
                ),
                row=2, col=1
            )

        # 4. 地址长度分布
        df['address_length'] = df['original'].fillna('').astype(str).str.len()
        fig.add_trace(
            go.Histogram(
                x=df['address_length'],
                nbinsx=50,
                name='地址长度',
                marker_color='lightcoral'
            ),
            row=2, col=2
        )

        # 5. 聚类质量指标
        if 'cluster_id' in df.columns:
            # 计算一些简单指标
            n_clusters = df['cluster_id'].nunique()
            avg_cluster_size = len(df) / n_clusters if n_clusters > 0 else 0

            fig.add_trace(
                go.Indicator(
                    mode="number+delta",
                    value=n_clusters,
                    title={"text": "聚类数量"},
                    domain={'row': 2, 'column': 0}
                ),
                row=3, col=1
            )

            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=avg_cluster_size,
                    title={"text": "平均聚类大小"},
                    domain={'row': 2, 'column': 1}
                ),
                row=3, col=1
            )

        # 6. 处理统计表
        stats_data = []
        if 'success' in df.columns:
            success_rate = df['success'].mean() * 100
            stats_data.append(['标准化成功率', f'{success_rate:.1f}%'])

        if 'latitude' in df.columns:
            geocoded_rate = df['latitude'].notna().mean() * 100
            stats_data.append(['地理编码成功率', f'{geocoded_rate:.1f}%'])

        if 'cluster_id' in df.columns:
            stats_data.append(['聚类数量', str(df['cluster_id'].nunique())])

        if stats_data:
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=['指标', '值'],
                        fill_color='paleturquoise',
                        align='left'
                    ),
                    cells=dict(
                        values=list(zip(*stats_data)),
                        fill_color='lavender',
                        align='left'
                    )
                ),
                row=3, col=2
            )

        # 更新布局
        fig.update_layout(
            height=1200,
            showlegend=False,
            title_text=f"地址聚类分析报告 - {output_prefix}",
            template='plotly_white'
        )

        # 保存HTML文件
        output_file = f"{output_prefix}_interactive_report.html"
        fig.write_html(output_file)
        print(f"交互式报告已保存: {output_file}")

        return fig

    def create_comprehensive_report(self, df: pd.DataFrame,
                                    similarity_df: pd.DataFrame = None,
                                    output_prefix: str = 'comprehensive_report'):
        """创建综合报告（包含所有图表）"""
        import os

        output_dir = f"{output_prefix}_visualization"
        os.makedirs(output_dir, exist_ok=True)

        print(f"📊 生成综合报告到目录: {output_dir}")

        # 1. 聚类分布图
        self.plot_cluster_distribution(
            df,
            os.path.join(output_dir, 'cluster_distribution.png')
        )

        # 2. 地理分布图
        self.plot_geographic_distribution(
            df,
            os.path.join(output_dir, 'geographic_distribution.png')
        )

        # 3. 相似度分布图（如果有）
        if similarity_df is not None:
            self.plot_similarity_distribution(
                similarity_df,
                os.path.join(output_dir, 'similarity_distribution.png')
            )

        # 4. 交互式地图
        self.create_interactive_map(
            df,
            os.path.join(output_dir, 'interactive_map.html')
        )

        # 5. 交互式报告
        self.create_interactive_report(
            df, similarity_df,
            os.path.join(output_dir, 'interactive_report')
        )

        # 6. 生成README
        self._generate_report_readme(output_dir, df, similarity_df)

        print(f"✅ 综合报告生成完成！")
        print(f"📁 所有文件保存在: {output_dir}")

        return output_dir

    def _generate_report_readme(self, output_dir: str, df: pd.DataFrame,
                                similarity_df: pd.DataFrame = None):
        """生成报告README文件"""
        readme_content = f"""
# 地址聚类分析报告

## 报告概述
生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
数据规模: {len(df):,} 条地址

## 文件说明
1. `cluster_distribution.png` - 聚类分布统计图
2. `geographic_distribution.png` - 地理分布图
3. `similarity_distribution.png` - 相似度分布图（如可用）
4. `interactive_map.html` - 交互式地图（使用浏览器打开）
5. `interactive_report.html` - 交互式分析报告（使用浏览器打开）

## 关键统计指标
- 地址总数: {len(df):,}
- 聚类数量: {df['cluster_id'].nunique() if 'cluster_id' in df.columns else 'N/A'}
- 标准化成功率: {df['success'].mean() * 100 if 'success' in df.columns else 'N/A':.1f}%
- 地理编码成功率: {df['latitude'].notna().mean() * 100 if 'latitude' in df.columns else 'N/A':.1f}%

## 使用说明
1. 查看静态图表: 直接打开PNG文件
2. 查看交互式地图: 用浏览器打开 `interactive_map.html`
3. 查看交互式报告: 用浏览器打开 `interactive_report.html`

## 数据处理配置
- 相似度阈值: 0.7
- 最大搜索半径: 50km
- 聚类算法: 带空间约束的连通分量算法
"""

        readme_file = os.path.join(output_dir, 'README.md')
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
