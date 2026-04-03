# 文献综述与外文翻译 Spec

## Why
用户需要完成本科毕业设计的文献综述与外文翻译任务。这两份文档是毕业设计的重要组成部分，分别用于梳理研究现状和学习前沿外文文献。

## What Changes
- 在 `Finance/04_项目文档/01_学术论文/reports/` 目录下生成两份 `.docx` 文档：
    1.  `文献综述_基于PI-KAN的金融时序相变探测研究.docx`
    2.  `外文翻译_KAN_Kolmogorov-Arnold_Networks.docx` (暂定翻译 KAN 原始论文或相关高引论文)

## Impact
- **Affected specs**: 无直接关联的其他 Spec。
- **Affected code**: 无代码变更，仅生成文档。

## ADDED Requirements
### Requirement: 文献综述生成
系统应根据开题报告内容，搜集并整理不少于 15 篇相关文献（包含至少 2 篇外文），撰写符合框架要求的文献综述。

#### Scenario: 撰写流程
- **WHEN** 用户请求开始撰写
- **THEN** 
    1.  搜集关于 PI-KAN, PINNs, KAN, 金融时间序列, 相变探测, 经济物理学 等主题的文献。
    2.  筛选出核心文献（如 KAN 原始论文, PINNs 金融应用论文, 经典经济物理学著作）。
    3.  按照“绪论-研究现状-总结-参考文献”的结构撰写综述。
    4.  生成 `.docx` 文件。

### Requirement: 外文翻译生成
系统应选取一篇高相关性的外文核心文献（推荐 *KAN: Kolmogorov-Arnold Networks* 或 *Physics-Informed Neural Networks in Finance*），进行不少于 3500 汉字的翻译。

#### Scenario: 翻译流程
- **WHEN** 确定翻译文献
- **THEN**
    1.  获取文献原文。
    2.  翻译摘要、引言、核心方法论等章节，确保字数达标。
    3.  按“左译文右原文”或分段对照格式排版。
    4.  生成 `.docx` 文件。
