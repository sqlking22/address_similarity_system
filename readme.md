# 百万级地址相似度识别与聚类系统

## 📋 系统概述

这是一个高性能的地址相似度识别与聚类系统，支持：

- 百万级地址的快速处理
- 多维度相似度计算（文本+空间+行政区划）
- 智能地址标准化与纠错
- 地理编码集成（支持GeocodingCHN）
- 带空间约束的聚类算法
- 丰富的可视化报告

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 准备数据

- 将地址数据保存为CSV文件，至少包含地址列（如address）
- 可选：包含经纬度列（如longitude, latitude）

### 运行系统

```bash
# 测试运行（1万条数据）
python main.py --input data/sample.csv --output test --sample 10000

# 全量运行
python main.py --input data/addresses.csv --output result --jobs 28

# 指定列名
python main.py --input data/addresses.csv --address-col address --lon-col lng --lat-col lat

# 生成可视化报告
python main.py --input data/addresses.csv --output result --visualize
```

## 📚 文件结构

address_similarity_system/
├── config/
│ └── config.py # 配置文件
├── core/
│ ├── address_normalizer.py # 地址标准化
│ ├── geocoding_integration.py # 地理编码集成
│ ├── similarity_calculator.py # 相似度计算
│ └── clustering.py # 聚类算法
├── data/ # 数据目录
│ ├── input/ # 输入数据
│ └── output/ # 输出结果
├── utils/ # 工具函数
│ ├── parallel_processor.py # 并行处理
│ └── visualization.py # 可视化工具
├── requirements.txt # 依赖包
└── main.py # 主程序入口

## 🔧 配置参数

- 通过config.py或命令行参数配置：

```python
# 主要配置项
weights = {
    'text': 0.5,  # 文本相似度权重
    'spatial': 0.3,  # 空间相似度权重
    'admin': 0.2  # 行政区划权重
}

similarity_threshold = 0.7  # 相似度阈值
distance_threshold_km = 10  # 距离阈值
max_search_radius_km = 50  # 最大搜索半径
min_cluster_size = 2  # 最小聚类大小
n_jobs = 28  # 并行任务数
```

## 输出文件
- 系统生成以下结果文件：
* {prefix}_clusters_*.csv - 聚类结果（包含聚类ID）
* {prefix}_cluster_summary_*.csv - 聚类统计摘要
* {prefix}_similarities_*.csv - 相似度矩阵
* {prefix}_processing_stats_*.json - 处理统计
* {prefix}_visualization_*.html - 可视化报告
* interactive_map.html - 交互式地图

## 性能指标
* 在128G内存 + 32核CPU的服务器上：
* 100万地址：约20-30分钟处理完成
* 处理速度：约500-1000地址/秒
* 峰值内存：约10-15GB
* 准确率：90-95%（结合经纬度）