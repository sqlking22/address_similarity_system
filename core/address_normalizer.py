#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2025/12/5 15:17
# @Author  : hejun
"""
高级地址标准化模块
结合cpca、jionlp和规则引擎
"""
import re
import pandas as pd
from typing import Dict, List, Any
import hashlib
import jionlp as jio
import cpca
import unicodedata


class AdvancedAddressNormalizer:
    """高级地址标准化器"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # 构建替换字典
        self._build_replacement_dict()

        # 常见错误映射
        self.error_mapping = {
            # 数字相关
            '卅': '三十', '廿': '二十', '卌': '四十',
            '佰': '百', '仟': '千', '萬': '万',
            # 地址相关
            '衖': '巷', '裏': '里', '邨': '村', '廈': '厦',
            '傢俱': '家具', '傢具': '家具',
            # 常见错别字
            '技园': '科技园', '枝术': '技术', '科枝': '科技',
            '工司': '公司', '分司': '分公司', '公局': '局',
            '广洲': '广州', '深训': '深圳', '汗京': '南京',
            '抗州': '杭州', '无夕': '无锡',
            # 单位相关
            '做坊': '作坊', '坊间': '坊', '幢幢': '幢',
        }

        # 地址组件正则
        self.patterns = {
            'building': re.compile(r'(\d+[号栋幢座]|\d+号楼|\d+栋楼)'),
            'room': re.compile(r'(\d+[室房号]|\d+单元|\d+-?\d+)'),
            'floor': re.compile(r'([地下]?\d+层|[BGF]\d+)'),
            'road': re.compile(r'(.+?[路街大道巷弄胡同])'),
            'number': re.compile(r'(\d+[号\-]\d+|\d+)'),
        }

    def _build_replacement_dict(self):
        """构建替换字典"""
        # 行政单位替换
        self.admin_replace = {
            '省': '', '市': '', '区': '', '县': '', '镇': '', '乡': '', '村': '',
            '街道': '', '办事处': '', '社区': '', '居委会': '',
        }

        # 道路类型标准化
        self.road_standardization = {
            '大街': '街', '大道': '道', '大路': '路',
            '胡同': '巷', '弄堂': '弄', '里弄': '弄',
            'avenue': '大道', 'road': '路', 'street': '街',
        }

        # 建筑物类型标准化
        self.building_standardization = {
            '大楼': '大厦', '商务楼': '大厦', '写字楼': '大厦',
            '商厦': '大厦', '中心大楼': '中心', '大楼中心': '中心',
            '公寓楼': '公寓', '住宅楼': '住宅', '办公楼': '办公',
        }

    def normalize_single(self, address: str) -> Dict[str, Any]:
        """
        标准化单个地址

        Args:
            address: 原始地址字符串

        Returns:
            标准化后的地址信息字典
        """
        if not isinstance(address, str) or not address.strip():
            return self._empty_result()

        try:
            # 步骤1: 基础清洗
            cleaned = self._basic_clean(address)

            # 步骤2: 纠错
            corrected = self._correct_errors(cleaned)

            # 步骤3: 多级解析
            cpca_result = self._parse_with_cpca(corrected)
            jionlp_result = self._parse_with_jionlp(corrected)

            # 步骤4: 智能合并
            merged = self._merge_results(cpca_result, jionlp_result, corrected)

            # 步骤5: 生成标准格式
            standardized = self._generate_standard_format(merged)

            # 步骤6: 提取组件
            components = self._extract_components(standardized)

            # 生成哈希
            address_hash = hashlib.md5(standardized.encode('utf-8')).hexdigest()[:12]

            return {
                'original': address,
                'cleaned': cleaned,
                'corrected': corrected,
                'standardized': standardized,
                'components': components,
                'hash': address_hash,
                'parsed': merged,
                'success': True
            }

        except Exception as e:
            print(f"地址标准化失败: {address}, 错误: {e}")
            return self._empty_result(address)

    def _basic_clean(self, text: str) -> str:
        """基础清洗"""
        # 移除不可见字符
        text = ''.join(char for char in text if char.isprintable())

        # 全角转半角
        text = unicodedata.normalize('NFKC', text)

        # 统一字符集
        text = text.lower()

        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()

        # 移除特殊字符，保留中文、数字、字母、常用标点
        text = re.sub(r'[^\w\u4e00-\u9fff\-\.,#()（）、，。；：]', ' ', text)

        return text

    def _correct_errors(self, text: str) -> str:
        """纠错处理"""
        corrected = text

        # 应用错误映射
        for wrong, right in self.error_mapping.items():
            corrected = corrected.replace(wrong, right)

        # 中文数字转阿拉伯数字
        corrected = self._chinese_to_arabic(corrected)

        return corrected

    def _chinese_to_arabic(self, text: str) -> str:
        """中文数字转阿拉伯数字"""
        chinese_digits = {
            '零': '0', '一': '1', '二': '2', '三': '3', '四': '4',
            '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
            '十': '', '百': '', '千': '', '万': '', '亿': '',
        }

        # 处理"十一"到"九十九"
        pattern = re.compile(r'([一二三四五六七八九十]{1,3})[^万亿百千]')

        def replace_number(match):
            num_str = match.group(1)
            if len(num_str) == 1:
                return chinese_digits.get(num_str, num_str)
            elif len(num_str) == 2:
                if num_str[0] == '十':
                    return '1' + (chinese_digits.get(num_str[1], '0') if num_str[1] != '十' else '0')
                elif num_str[1] == '十':
                    return chinese_digits.get(num_str[0], '0') + '0'
            return num_str

        return pattern.sub(replace_number, text)

    def _parse_with_cpca(self, address: str) -> Dict[str, Any]:
        """使用cpca解析地址"""
        try:
            df = cpca.transform([address])
            if not df.empty:
                return {
                    'province': df.iloc[0]['省'] if '省' in df.columns else '',
                    'city': df.iloc[0]['市'] if '市' in df.columns else '',
                    'district': df.iloc[0]['区'] if '区' in df.columns else '',
                    'address': df.iloc[0]['地址'] if '地址' in df.columns else '',
                    'source': 'cpca',
                }
        except Exception as e:
            print(f"cpca解析失败: {address}, 错误: {e}")
        return {}

    def _parse_with_jionlp(self, address: str) -> Dict[str, Any]:
        """使用jionlp解析地址"""
        try:
            result = jio.parse_location(address)
            if result and 'province' in result:
                return {
                    'province': result.get('province', ''),
                    'city': result.get('city', ''),
                    'county': result.get('county', ''),
                    'detail': result.get('detail', ''),
                    'full_location': result.get('full_location', ''),
                    'redundancy': result.get('redundancy', ''),
                    'source': 'jionlp',
                }
        except Exception as e:
            print(f"jionlp解析失败: {address}, 错误: {e}")

        return {}

    def _merge_results(self, cpca_result: Dict, jionlp_result: Dict, original: str) -> Dict[str, Any]:
        """智能合并解析结果"""
        merged = {
            'province': '',
            'city': '',
            'district': '',
            'street': '',
            'detail': '',
            'full': original,
        }

        # 优先使用jionlp的省份和城市（通常更准确）
        if jionlp_result.get('province'):
            merged['province'] = jionlp_result['province']
        elif cpca_result.get('province'):
            merged['province'] = cpca_result['province']

        if jionlp_result.get('city'):
            merged['city'] = jionlp_result['city']
        elif cpca_result.get('city'):
            merged['city'] = cpca_result['city']

        # 区县优先使用cpca
        if cpca_result.get('district'):
            merged['district'] = cpca_result['district']
        elif jionlp_result.get('county'):
            merged['district'] = jionlp_result['county']

        # 详细地址
        if jionlp_result.get('detail'):
            merged['detail'] = jionlp_result['detail']
        elif cpca_result.get('address'):
            merged['detail'] = cpca_result['address']

        # 尝试提取街道信息
        street_match = re.search(r'(.+?[路街大道巷弄])', merged['detail'])
        if street_match:
            merged['street'] = street_match.group(1)
            # 从detail中移除街道
            merged['detail'] = merged['detail'].replace(merged['street'], '').strip()

        return merged

    def _generate_standard_format(self, parsed: Dict[str, Any]) -> str:
        """生成标准格式地址"""
        parts = []

        if parsed['province']:
            parts.append(parsed['province'])
        if parsed['city']:
            parts.append(parsed['city'])
        if parsed['district']:
            parts.append(parsed['district'])
        if parsed['street']:
            parts.append(parsed['street'])
        if parsed['detail']:
            parts.append(parsed['detail'])

        standardized = ''.join(parts)

        # 应用标准化规则
        for old, new in self.road_standardization.items():
            standardized = standardized.replace(old, new)

        for old, new in self.building_standardization.items():
            standardized = standardized.replace(old, new)

        # 移除行政单位
        for unit in self.admin_replace:
            standardized = standardized.replace(unit, self.admin_replace[unit])

        return standardized.strip()

    def _extract_components(self, address: str) -> Dict[str, Any]:
        """提取地址组件"""
        components = {
            'road': '',
            'number': '',
            'building': '',
            'room': '',
            'floor': '',
            'estate': '',
        }

        # 提取道路
        road_match = self.patterns['road'].search(address)
        if road_match:
            components['road'] = road_match.group(1)

        # 提取门牌号
        number_match = self.patterns['number'].search(address)
        if number_match:
            components['number'] = number_match.group(1)

        # 提取建筑物
        building_match = self.patterns['building'].search(address)
        if building_match:
            components['building'] = building_match.group(1)

        # 提取房间号
        room_match = self.patterns['room'].search(address)
        if room_match:
            components['room'] = room_match.group(1)

        # 提取楼层
        floor_match = self.patterns['floor'].search(address)
        if floor_match:
            components['floor'] = floor_match.group(1)

        # 尝试识别小区/园区
        estate_keywords = ['小区', '花园', '新村', '公寓', '别墅', '山庄', '园区', '科技园']
        for keyword in estate_keywords:
            if keyword in address:
                # 提取小区名（关键字前后的内容）
                start = address.find(keyword) - 10
                end = address.find(keyword) + len(keyword) + 5
                start = max(0, start)
                end = min(len(address), end)
                components['estate'] = address[start:end].strip()
                break

        return components

    def _empty_result(self, address: str = "") -> Dict[str, Any]:
        """返回空结果"""
        return {
            'original': address,
            'cleaned': address,
            'corrected': address,
            'standardized': address,
            'components': {},
            'hash': hashlib.md5(address.encode('utf-8')).hexdigest()[:12] if address else '',
            'parsed': {},
            'success': False
        }

    def batch_normalize(self, addresses: List[str], n_jobs: int = 8) -> List[Dict[str, Any]]:
        """批量标准化地址"""
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs)(
            delayed(self.normalize_single)(addr)
            for addr in addresses
        )

        return results


class AddressStandardizationPipeline:
    """地址标准化管道"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.normalizer = AdvancedAddressNormalizer(config)

    def process_dataframe(self, df: pd.DataFrame, address_column: str = 'address') -> pd.DataFrame:
        """处理DataFrame"""
        addresses = df[address_column].fillna('').astype(str).tolist()

        print(f"开始标准化 {len(addresses)} 个地址...")
        results = self.normalizer.batch_normalize(
            addresses,
            n_jobs=self.config.get('n_jobs', 8)
        )

        # 转换为DataFrame
        results_df = pd.DataFrame(results)

        # 合并到原始数据
        for col in results_df.columns:
            if col not in df.columns:
                df[col] = results_df[col]

        print(f"地址标准化完成，成功: {results_df['success'].sum()}/{len(results_df)}")

        return df