#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2025/12/5 15:35
# @Author  : hejun
"""
地址相似度识别与聚类系统
"""

from core.address_normalizer import AddressStandardizationPipeline, AdvancedAddressNormalizer
from core.geocoding_integration import GeocodingCHNIntegration, MultiSourceGeocoder
from core.similarity_calculator import MultiDimensionalSimilarityCalculator
from core.clustering import AddressClustering
from utils.parallel_processor import ParallelProcessor, MemoryOptimizedProcessor, SimilarityMatrixBuilder
from utils.visualization import VisualizationTools

__version__ = '1.0.0'
__author__ = 'Address Similarity System'

__all__ = [
    'AddressStandardizationPipeline',
    'AdvancedAddressNormalizer',
    'GeocodingCHNIntegration',
    'MultiSourceGeocoder',
    'MultiDimensionalSimilarityCalculator',
    'AddressClustering',
    'ParallelProcessor',
    'MemoryOptimizedProcessor',
    'SimilarityMatrixBuilder',
    'VisualizationTools'
]