#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2025/12/5 15:36
# @Author  : hejun
"""
项目安装文件
"""
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# 读取README
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="address-similarity-system",
    version="1.0.0",
    description="百万级地址相似度识别与聚类系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Address Similarity Team",
    author_email="example@example.com",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="address, similarity, clustering, geocoding, chinese",
    packages=find_packages(where="."),
    python_requires=">=3.8, <4",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.2.0",
        "scipy>=1.10.0",
        "geopy>=2.3.0",
        "rapidfuzz>=3.0.0",
        "python-Levenshtein>=0.21.0",
        "jaro-winkler>=0.1.0",
        "datasketch>=1.5.0",
        "cpca>=0.2.0",
        "jionlp>=1.4.12",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "folium>=0.14.0",
        "plotly>=5.14.0",
        "joblib>=1.2.0",
        "numba>=0.57.0",
        "polars>=0.18.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "black>=22.0", "flake8>=5.0"],
        "gpu": ["cupy-cuda11x>=11.0"],
        "spark": ["pyspark>=3.3.0"],
    },
    entry_points={
        "console_scripts": [
            "address-similarity=main:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/example/address-similarity/issues",
        "Source": "https://github.com/example/address-similarity",
    },
)