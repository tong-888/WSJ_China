# WSJ China: 基于华尔街日报新闻的中国经济主题分析

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

本项目旨在通过自然语言处理和机器学习技术，从海量华尔街日报（Wall Street Journal）新闻中筛选、分析与中国相关的经济新闻，并使用监督式主题模型（sLDA）挖掘能够解释和预测人民币汇率波动的潜在经济叙事主题。

## 项目概述

本项目构建了一个完整的数据处理流水线，从原始新闻数据开始，经过多阶段筛选和清洗，最终生成可用于监督式主题建模的高质量文本数据集。整个流程处理了超过150万篇全球新闻，最终筛选出约18万篇与中国高度相关的新闻。

### 核心目标

- 从海量英文新闻中精准筛选与中国政治、经济、科技等领域相关的新闻
- 通过实体短语固化和深度文本清洗，构建高质量的文本语料库
- 应用监督式主题模型（sLDA）发现与人民币汇率波动相关的经济叙事主题
- 为经济政策分析和金融预测提供数据支持和洞见

## 项目架构

项目的整体处理流程遵循"漏斗模型"，从海量原始数据开始，通过层层筛选和净化，最终得到高度浓缩的核心数据：

![](/reports/data-2025-08-08-1048.png "数据处理过程")

### 各阶段详细说明

1. **智能中国新闻筛选**
   - 基于分级关键词体系（1-5级相关性）的两阶段筛选
   - 第一阶段使用Flashtext进行高效初筛
   - 第二阶段使用spaCy进行上下文精筛

2. **实体与短语固化**
   - 通过算法发现和人工审核相结合的方式识别重要短语
   - 使用Aho-Corasick算法进行高性能文本替换
   - 将多词概念固化为不可分割的"超级词"

3. **深度文本清洗**
   - 词性过滤，只保留名词、动词、形容词等有意义的词性
   - 定制化停用词处理（spaCy默认 + 人工审核 + 项目专属）
   - 词形还原和标准化处理

4. **监督式主题建模（sLDA）**
   - 将清洗后的文本与人民币汇率时间序列数据对齐
   - 训练sLDA模型发现与汇率波动相关的主题
   - 进行主题解读和经济叙事归因分析

### 脚本说明
1. **Jupyter Notebook脚本**
     1. `scripts/01_China_News_Filtering.ipynb` - 筛选中国相关新闻
     2. `scripts/02_Entity_Phrase_Solidification.ipynb` - 实体短语发现
     3. `scripts/02_B_Phrase_Integration.ipynb` - 短语规则整合
     4. `scripts/02_C_Phrase_Application.ipynb` - 短语固化应用
     5. `scripts/03_Deep_Text_Cleaning.ipynb` - 深度文本清洗

2. **sLDA建模**
   - 准备汇率时间序列数据
   - 运行sLDA建模脚本（待开发）

## 项目结构

```
WSJ_China/
├── configs/                            # 配置文件
│   └── china_keywords_collection.json  # 中国相关关键词集合
│   └── expert_rules.csv                # 自定义短语规则
│   └── project_specific_stopwords.txt  # 自定义停用词
├── reports/                            # 需求分析和文档
│   ├── 总流程.md                        # 总体流程规划
│   ├── 数据文件介绍.md                   # 数据文件说明
│   └── 需求分析_*.md                    # 各阶段需求分析
├── scripts/                            # 处理脚本
│   ├── 01_China_News_Filtering.ipynb
│   ├── 02_Entity_Phrase_Solidification.ipynb
│   ├── 02_B_Phrase_Integration.ipynb
│   ├── 02_C_Phrase_Application.ipynb
│   └── 03_Deep_Text_Cleaning.ipynb
└── data-processed/                     # 数据文件（文件过大不提供）
```

## 致谢
感谢上海大学 龚玉婷教授

本项目在开发过程中使用了以下开源库：
- [spaCy](https://spacy.io/) - 工业级自然语言处理库
- [Gensim](https://radimrehurek.com/gensim/) - 主题建模和文档相似性分析库
- [pandas](https://pandas.pydata.org/) - 数据操作和分析库
- [Flashtext](https://github.com/vi3k6i5/flashtext) - 高性能关键字搜索和替换库