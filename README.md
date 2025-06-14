

# RAG_pdf

可上传pdf文档的RAG开发项目

# 要求和文献

https://github.com/opendatalab/MinerU
https://github.com/weiwill88/Local_Pdf_Chat_RAG
https://github.com/hhy-huang/HiRAG
以这本pdf作为知识库，要求使用RAG技术，写一个问答agent。agent能够针对这本pdf中的细节问题（例如其中的数字、项目介绍等信息）进行准确的回答

给几个测试样例：
1. 大丰电子信息产业园污水处理工程，一期工程的处理水量是多少？
11000 m3/d

2. 高浓度有机废水好氧 MBR 法生物处理技术的处理范围包含豆类吗？
不包含

3. 请帮我找到国家环境保护工业废水污染控制工程技术中心负责人的邮箱
13911007910@163.com


# 基础RAG框架

## 数据集构建
https://zhuanlan.zhihu.com/p/13467487568
使用的这篇教程进行提示词微调生成的数据集。（manus生成）

共56组QA对
进行测试，迭代检索为3轮次

由4omini作为答案评价者给出correct or incorrect
最终正确率为**76%**（56总样本)

# 检索优化

## 优化事项1：
优化迭代搜索，query生成部分提示词，尝试使用短关键词进行RAG检索，发现出现不同的问题
首先，两次回答正确的题目分布不同，短关键词更擅长查找具体数字，数目，邮箱号码等，正确率降低到67%
## 优化事项2：
更换基座模型，尝试了使用deepseekV3来作为基座模型，正确率提升至82%
## 优化事项3：
首先，该文档有明显的分段特征，一个查询（示例）只关注到一个具体项目的文字，但是一个具体项目，该项目专有名词可能会和需要的信息差距甚远（例如江水公司 确定 项目编号，资源成本内容在该项目的末尾部分，是所有项目的通用章节，普通的RAG只靠相似度进行搜索无法准确搜集）。
### 方向1：
本人先尝试在原RAG基础上进行改动，使用MinerU将pdf转为Markdown文档，目的是为了获取结构化大纲信息，根据标题级数来实现层级化搜索。检索流程如下：

- MinerU将pdf转为markdown，根据标题级数分为数个子文档，（1-N）
- 得到用户Query
- 整体pdf构建知识库，Query作为init输入进行检索，得到前三chunk
- 追踪chunk对应的子文档i
- 对于追踪到的子文档{i_1.....i_n},分别在它们的知识库上进行迭代搜索，最后综合得到答案。
  这样的结构很适合对于项目书这种合集类文档的查找，避免两个章节之间引起链接导致检索错误。
  经过测试，模型的准确率达到85%



### 方向2：

查阅资料，目前RAG在现有的基础上做了很多数据结构的改动，例如GraphRAG和HiRAG，都是以图或层结构来反映知识关系，实现关联查找。
本人复现了HiRAG项目，先将大致过程结果汇报：
使用deepseekV3作为基座模型
构建知识库需要调用相当多的次数（该pdf大致花费的deeepseekV3 3元，白天收费档）
构建完成后查询很迅速，只需要一次调用api
准确率**98.21%**（55/56）

# 后期可以扩充的功能

- 记忆搜索功能
- QA对存储，将数据集部分存入RAG中作为知识库 转成markdown格式

# How to use

```bash
# basic RAG environment
cd Local_Pdf_Chat_RAG
conda create -n RAG python=3.10
pip install -r requirements.txt
# HiRAG environment
cd HiRAG 
conda create -n HiRAG python=3.10
pip install -e .
```

# 目录

```bash
RAG/
├── .git/                   # Git版本控制目录
├── HiRAG/                  # 主要RAG实现项目
├── Local_Pdf_Chat_RAG/     # 本地PDF文档聊天RAG实现
├── evaluation/             # 评估相关代码和结果
├── dataset/                # 数据集目录
├── Scripts/                # 脚本工具目录
└── README.md               # 项目根目录说明文档

HiRAG/
├── config.yaml              # 配置文件
├── requirements.txt         # 项目依赖
├── setup.py                # 项目安装配置
├── readme.md               # 项目说明文档
├── LICENSE                 # 许可证文件
├── hi_Search_deepseek_batch_qa.py  # DeepSeek模型批处理QA实现
├── hi_Search_glm.py        # GLM模型搜索实现
├── hi_Search_openai.py     # OpenAI模型搜索实现
├── evaluation_results_1.json  # 评估结果
├── qa_results/             # QA结果存储目录
├── hirag/                  # 核心代码目录
├── eval/                   # 评估相关代码
├── docs/                   # 文档目录
├── case/                   # 案例目录
├── imgs/                   # 图片资源目录
├── 1.txt/                  # 知识文档txt版，和dataset文件夹里的相同
└── your_work_dir/          # 工作目录

Local_Pdf_Chat_RAG/
├── rag_demo_pro.py         # 主要RAG演示程序
├── batch_qa_json.py        # 批处理QA实现
├── lasj.py                 # 工具脚本
├── api_router.py           # API路由实现
├── requirements.txt        # 项目依赖
├── example.env             # 环境变量示例
├── utils/                  # 工具函数目录
├── images/                 # 图片资源目录
└── qa_results/             # QA结果存储目录

evaluation/
├── lasj.py                 # 评估工具脚本
├── evaluation_results_basic.json           # 基础评估结果
├── evaluation_results_use_shorter_query.json  # 短查询评估结果
└── evaluation_results_HiRAG.json           # HiRAG评估结果

dataset/
├── knowledge.txt           # 知识库文本文件
├── knowledge.pdf           # 知识库PDF文件
├── knowledge.md            # 知识库Markdown文件
├── qa_dataset.json         # QA数据集
└── qa_dataset_change_doubao.json  # 修改后的QA数据集

Scripts/
├── merge_json.py           # JSON合并工具
└── merged_results/         # 合并结果存储目录
```

# 运行RAG项目

```
cd Local_Pdf_Chat_RAG
conda activate RAG
# 在rag_demo_pro.py 的call_siliconflow_api函数中配置api
python rag_demo_pro.py --file ./1.pdf #单个query测试
python batch_qa_json.py # 数据集测试，需要修改路径

# 在config.yml里配置api
cd HiRAG
conda activate HiRAG
python hi_Search_deepseek_batch_qa.py# 数据集测试，需要修改路径

# 测试
cd evaluation
python lasj.py #修改路径和api，Openai函数可能需要重写，使用的第三方中转站
```

# Disclaimer

This project is provided "as is" without any warranty. Use it at your own risk.
