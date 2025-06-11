import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
from pdfminer.high_level import extract_text_to_fp
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import faiss
import requests
import json
from io import StringIO
from langchain_text_splitters import RecursiveCharacterTextSplitter
import socket
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from datetime import datetime
import hashlib
import re
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import numpy as np
import jieba
import threading
from functools import lru_cache

# 加载环境变量
load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SEARCH_ENGINE = "google"
RERANK_METHOD = os.getenv("RERANK_METHOD", "cross_encoder")
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_API_KEY = ""
SILICONFLOW_API_URL = os.getenv("SILICONFLOW_API_URL", "https://api.siliconflow.cn/v1/chat/completions")

# 配置请求重试
requests.adapters.DEFAULT_RETRIES = 3
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

# 初始化组件
EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# FAISS相关全局变量
faiss_index = None
faiss_contents_map = {}  # original_id -> content
faiss_metadatas_map = {} # original_id -> metadata
faiss_id_order_for_index = []

# 初始化交叉编码器
cross_encoder = None
cross_encoder_lock = threading.Lock()

class BM25IndexManager:
    def __init__(self):
        self.bm25_index = None
        self.doc_mapping = {}  # 映射BM25索引位置到文档ID
        self.tokenized_corpus = []
        self.raw_corpus = []
        
    def build_index(self, documents, doc_ids):
        """构建BM25索引"""
        self.raw_corpus = documents
        self.doc_mapping = {i: doc_id for i, doc_id in enumerate(doc_ids)}
        
        # 对文档进行分词，使用jieba分词器更适合中文
        self.tokenized_corpus = []
        for doc in documents:
            # 对中文文档进行分词
            tokens = list(jieba.cut(doc))
            self.tokenized_corpus.append(tokens)
        
        # 创建BM25索引
        self.bm25_index = BM25Okapi(self.tokenized_corpus)
        return True
        
    def search(self, query, top_k=5):
        """使用BM25检索相关文档"""
        if not self.bm25_index:
            return []
        
        # 对查询进行分词
        tokenized_query = list(jieba.cut(query))
        
        # 获取BM25得分
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        
        # 获取得分最高的文档索引
        top_indices = np.argsort(bm25_scores)[-top_k:][::-1]
        
        # 返回结果
        results = []
        for idx in top_indices:
            if bm25_scores[idx] > 0:  # 只返回有相关性的结果
                results.append({
                    'id': self.doc_mapping[idx],
                    'score': float(bm25_scores[idx]),
                    'content': self.raw_corpus[idx]
                })
        
        return results
    
    def clear(self):
        """清空索引"""
        self.bm25_index = None
        self.doc_mapping = {}
        self.tokenized_corpus = []
        self.raw_corpus = []

# 初始化BM25索引管理器
BM25_MANAGER = BM25IndexManager()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_cross_encoder():
    """延迟加载交叉编码器模型"""
    global cross_encoder
    if cross_encoder is None:
        with cross_encoder_lock:
            if cross_encoder is None:
                try:
                    # 使用多语言交叉编码器，更适合中文
                    cross_encoder = CrossEncoder('sentence-transformers/distiluse-base-multilingual-cased-v2')
                    logging.info("交叉编码器加载成功")
                except Exception as e:
                    logging.error(f"加载交叉编码器失败: {str(e)}")
                    # 设置为None，下次调用会重试
                    cross_encoder = None
    return cross_encoder

def recursive_retrieval(initial_query, max_iterations=3, enable_web_search=False, model_choice="ollama"):
    """
    实现递归检索与迭代查询功能
    通过分析当前查询结果，确定是否需要进一步查询
    
    Args:
        initial_query: 初始查询
        max_iterations: 最大迭代次数
        enable_web_search: 是否启用网络搜索
        model_choice: 使用的模型选择("ollama"或"siliconflow")
        
    Returns:
        包含所有检索内容的列表
    """
    query = initial_query
    all_contexts = []
    all_doc_ids = []  # 使用原始ID
    all_metadata = []
    
    global faiss_index, faiss_contents_map, faiss_metadatas_map, faiss_id_order_for_index
    
    for i in range(max_iterations):
        logging.info(f"递归检索迭代 {i+1}/{max_iterations}，当前查询: {query}")
        
        web_results_texts = [] # Store text from web results for context building
        if enable_web_search and check_serpapi_key():
            try:
                # update_web_results now needs to handle FAISS directly or be adapted
                # For now, let's assume it returns texts to be added to context
                web_search_raw_results = update_web_results(query) # This function needs to be adapted for FAISS
                for res in web_search_raw_results:
                    text = f"标题：{res.get('title', '')}\\n摘要：{res.get('snippet', '')}"
                    web_results_texts.append(text)
                    # We would also need to add these to faiss_index, faiss_contents_map etc.
                    # and get their FAISS indices if we want them to be part of semantic search.
                    # This part is complex due to dynamic addition and potential ID clashes.
                    # For now, web results are added as pure text context, not searched semantically *again* within this loop's FAISS query.
            except Exception as e:
                logging.error(f"网络搜索错误: {str(e)}")
        
        query_embedding = EMBED_MODEL.encode([query])
        query_embedding_np = np.array(query_embedding).astype('float32')
        
        semantic_results_docs = []
        semantic_results_metadatas = []
        semantic_results_ids = []

        if faiss_index and faiss_index.ntotal > 0:
            try:
                D, I = faiss_index.search(query_embedding_np, k=10) # D: distances, I: indices
                # I contains the internal FAISS indices. We need to map them back to original IDs.
                for faiss_idx in I[0]: # I[0] because query_embedding_np was a batch of 1
                    if faiss_idx != -1 and faiss_idx < len(faiss_id_order_for_index):
                        original_id = faiss_id_order_for_index[faiss_idx]
                        semantic_results_docs.append(faiss_contents_map.get(original_id, ""))
                        semantic_results_metadatas.append(faiss_metadatas_map.get(original_id, {}))
                        semantic_results_ids.append(original_id)
            except Exception as e:
                logging.error(f"FAISS 检索错误: {str(e)}")
        
        bm25_results = BM25_MANAGER.search(query, top_k=10) # BM25_MANAGER.search returns list of dicts
        
        # Adapt hybrid_merge to work with current data structures
        # It expects semantic_results in a specific format if we pass it directly
        # For now, prepare a structure similar to old semantic_results for hybrid_merge
        prepared_semantic_results_for_hybrid = {
            "ids": [semantic_results_ids],
            "documents": [semantic_results_docs],
            "metadatas": [semantic_results_metadatas]
        }

        hybrid_results = hybrid_merge(prepared_semantic_results_for_hybrid, bm25_results, alpha=0.7)
        
        doc_ids_current_iter = []
        docs_current_iter = []
        metadata_list_current_iter = []
        
        if hybrid_results:
            for doc_id, result_data in hybrid_results[:10]: # doc_id here is the original_id
                doc_ids_current_iter.append(doc_id)
                docs_current_iter.append(result_data['content'])
                metadata_list_current_iter.append(result_data['metadata'])
        
        if docs_current_iter:
            try:
                reranked_results = rerank_results(query, docs_current_iter, doc_ids_current_iter, metadata_list_current_iter, top_k=5)
            except Exception as e:
                logging.error(f"重排序错误: {str(e)}")
                reranked_results = [(doc_id, {'content': doc, 'metadata': meta, 'score': 1.0}) 
                                  for doc_id, doc, meta in zip(doc_ids_current_iter, docs_current_iter, metadata_list_current_iter)]
        else:
            reranked_results = []
        
        current_contexts_for_llm = web_results_texts[:] # Start with web results for LLM context
        for doc_id, result_data in reranked_results:
            doc = result_data['content']
            metadata = result_data['metadata']
            
            if doc_id not in all_doc_ids:  
                all_doc_ids.append(doc_id)
                all_contexts.append(doc)
                all_metadata.append(metadata)
            current_contexts_for_llm.append(doc) # Add reranked local docs for LLM context
        
        if i == max_iterations - 1:
            break
            
        if current_contexts_for_llm: # Use combined web and local context for deciding next query
            current_summary = "\\n".join(current_contexts_for_llm[:3]) if current_contexts_for_llm else "未找到相关信息"
            
            next_query_prompt = f"""基于原始问题: {initial_query}
以及已检索信息: 
{current_summary}

分析是否需要进一步查询。如果需要，请提供新的查询问题，使用不同角度或更具体的关键词。
如果已经有充分信息，请回复'不需要进一步查询'。

新查询(如果需要):"""
            
            try:
                if model_choice == "siliconflow":
                    logging.info("使用SiliconFlow API分析是否需要进一步查询")
                    next_query_result = call_siliconflow_api(next_query_prompt, temperature=0.7, max_tokens=256)
                    # SiliconFlow API返回格式包含回答和可能的思维链，这里只需要回答部分来判断是否继续
                    # 假设call_siliconflow_api返回的是一个元组 (回答, 思维链) 或只是回答字符串
                    if isinstance(next_query_result, tuple):
                         next_query = next_query_result[0].strip() # 取回答部分
                    else:
                         next_query = next_query_result.strip() # 如果只返回字符串

                    # 移除潜在的思维链标记
                    if "<think>" in next_query:
                        next_query = next_query.split("<think>")[0].strip()


                else:
                    logging.info("使用本地Ollama模型分析是否需要进一步查询")
                    response = session.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": "deepseek-r1:1.5b",
                            "prompt": next_query_prompt,
                            "stream": False
                        },
                        timeout=30
                    )
                    # Ollama 返回格式不同，需要根据实际情况提取
                    next_query = response.json().get("response", "").strip()
                
                if "不需要" in next_query or "不需要进一步查询" in next_query or len(next_query) < 5:
                    logging.info("LLM判断不需要进一步查询，结束递归检索")
                    break
                    
                # 使用新查询继续迭代
                query = next_query
                logging.info(f"生成新查询: {query}")
            except Exception as e:
                logging.error(f"生成新查询时出错: {str(e)}")
                break
        else:
            # 如果当前迭代没有检索到内容，结束迭代
            break
    
    return all_contexts, all_doc_ids, all_metadata

def update_web_results(query: str, num_results: int = 5) -> list:
    """
    基于 SerpAPI 搜索结果。注意：此版本不将结果存入FAISS。
    它仅返回原始搜索结果。
    """
    results = serpapi_search(query, num_results)
    if not results:
        logging.info("网络搜索没有返回结果或发生错误")
        return []
    
    # 之前这里有删除旧网络结果和添加到ChromaDB的逻辑。
    # 由于FAISS IndexFlatL2不支持按ID删除，并且动态添加涉及复杂ID管理，
    # 此简化版本不将网络结果添加到FAISS索引。
    # 返回原始结果，供调用者决定如何使用（例如，仅作为文本上下文）。
    logging.info(f"网络搜索返回 {len(results)} 条结果，这些结果不会被添加到FAISS索引中。")
    return results # 返回原始SerpAPI结果列表

# 检查是否配置了SERPAPI_KEY
def check_serpapi_key():
    """检查是否配置了SERPAPI_KEY"""
    return SERPAPI_KEY is not None and SERPAPI_KEY.strip() != ""

# 添加文件处理状态跟踪
class FileProcessor:
    def __init__(self):
        self.processed_files = {}  # 存储已处理文件的状态
        
    def clear_files(self):
        """清空所有文件记录"""
        self.processed_files = {}
        
    def add_file(self, file_name):
        self.processed_files[file_name] = {
            'status': '等待处理',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'chunks': 0
        }
        
    def update_status(self, file_name, status, chunks=None):
        if file_name in self.processed_files:
            self.processed_files[file_name]['status'] = status
            if chunks is not None:
                self.processed_files[file_name]['chunks'] = chunks
                
    def get_file_list(self):
        return [
            f"📄 {fname} | {info['status']}"
            for fname, info in self.processed_files.items()
        ]

file_processor = FileProcessor()

#########################################
# 矛盾检测函数
#########################################
def detect_conflicts(sources):
    """精准矛盾检测算法"""
    key_facts = {}
    for item in sources:
        facts = extract_facts(item['text'] if 'text' in item else item.get('excerpt', ''))
        for fact, value in facts.items():
            if fact in key_facts:
                if key_facts[fact] != value:
                    return True
            else:
                key_facts[fact] = value
    return False

def extract_facts(text):
    """从文本提取关键事实（示例逻辑）"""
    facts = {}
    # 提取数值型事实
    numbers = re.findall(r'\b\d{4}年|\b\d+%', text)
    if numbers:
        facts['关键数值'] = numbers
    # 提取技术术语
    if "产业图谱" in text:
        facts['技术方法'] = list(set(re.findall(r'[A-Za-z]+模型|[A-Z]{2,}算法', text)))
    return facts

def evaluate_source_credibility(source):
    """评估来源可信度"""
    credibility_scores = {
        "gov.cn": 0.9,
        "edu.cn": 0.85,
        "weixin": 0.7,
        "zhihu": 0.6,
        "baidu": 0.5
    }
    
    url = source.get('url', '')
    if not url:
        return 0.5  # 默认中等可信度
    
    domain_match = re.search(r'//([^/]+)', url)
    if not domain_match:
        return 0.5
    
    domain = domain_match.group(1)
    
    # 检查是否匹配任何已知域名
    for known_domain, score in credibility_scores.items():
        if known_domain in domain:
            return score
    
    return 0.5  # 默认中等可信度

def extract_text(filepath):
    """改进的PDF文本提取方法"""
    output = StringIO()
    with open(filepath, 'rb') as file:
        extract_text_to_fp(file, output)
    return output.getvalue()


from pathlib import Path
def process_multiple_pdfs(files):
    """处理多个PDF文件"""
    if not files:
        return "请选择要上传的PDF文件", []
    
    try:
        # 清空向量数据库和相关存储
        # progress(0.1, desc="清理历史数据...")
        global faiss_index, faiss_contents_map, faiss_metadatas_map, faiss_id_order_for_index
        faiss_index = None
        faiss_contents_map = {}
        faiss_metadatas_map = {}
        faiss_id_order_for_index = []
        
        # 清空BM25索引
        BM25_MANAGER.clear()
        logging.info("成功清理历史FAISS数据和BM25索引")
        
        # 清空文件处理状态
        file_processor.clear_files()
        
        total_files = len(files)
        processed_results = []
        total_chunks = 0
        
        all_new_chunks = []
        all_new_metadatas = []
        all_new_original_ids = []
        
        for idx, file in enumerate(files, 1):
            pdf_path = Path(file)
            file_name = pdf_path.name
            try:
                # file_name = os.path.basename(file.name)
                # progress((idx-1)/total_files, desc=f"处理文件 {idx}/{total_files}: {file_name}")
                
                file_processor.add_file(file_name)
                text = extract_text(str(pdf_path))
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=400,        
                    chunk_overlap=40,     
                    separators=["\n\n", "\n", "。", "，", "；", "：", " ", ""]
                )
                chunks = text_splitter.split_text(text)
                
                if not chunks:
                    raise ValueError("文档内容为空或无法提取文本")
                
                doc_id = f"doc_{int(time.time())}_{idx}"
                
                # Store chunks and metadatas temporarily before batch embedding
                current_file_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
                current_file_metadatas = [{"source": file_name, "doc_id": doc_id} for _ in chunks]

                all_new_chunks.extend(chunks)
                all_new_metadatas.extend(current_file_metadatas)
                all_new_original_ids.extend(current_file_ids)
                
                total_chunks += len(chunks)
                file_processor.update_status(file_name, "处理完成", len(chunks))
                processed_results.append(f"✅ {file_name}: 成功处理 {len(chunks)} 个文本块")
                
            except Exception as e:
                error_msg = str(e)
                logging.error(f"处理文件 {file_name} 时出错: {error_msg}")
                file_processor.update_status(file_name, f"处理失败: {error_msg}")
                processed_results.append(f"❌ {file_name}: 处理失败 - {error_msg}")
        
        if all_new_chunks:
            # progress(0.8, desc="生成文本嵌入...")
            embeddings = EMBED_MODEL.encode(all_new_chunks, show_progress_bar=True)
            embeddings_np = np.array(embeddings).astype('float32')
            
            # progress(0.9, desc="构建FAISS索引...")
            if faiss_index is None: # Should always be None here due to clearing
                dimension = embeddings_np.shape[1]
                faiss_index = faiss.IndexFlatL2(dimension)
            
            faiss_index.add(embeddings_np)
            
            for i, original_id in enumerate(all_new_original_ids):
                faiss_contents_map[original_id] = all_new_chunks[i]
                faiss_metadatas_map[original_id] = all_new_metadatas[i]
            faiss_id_order_for_index.extend(all_new_original_ids) # Keep track of order for FAISS indices
            logging.info(f"FAISS索引构建完成，共索引 {faiss_index.ntotal} 个文本块")

        summary = f"\n总计处理 {total_files} 个文件，{total_chunks} 个文本块"
        processed_results.append(summary)
        
        # progress(0.95, desc="构建BM25检索索引...")
        update_bm25_index() # This will need to use faiss_contents_map
        
        file_list = file_processor.get_file_list()
        
        return "\n".join(processed_results), file_list
        
    except Exception as e:
        error_msg = str(e)
        logging.error(f"整体处理过程出错: {error_msg}")
        return f"处理过程出错: {error_msg}", []

# 新增：交叉编码器重排序函数
def rerank_with_cross_encoder(query, docs, doc_ids, metadata_list, top_k=5):
    """
    使用交叉编码器对检索结果进行重排序
    
    参数:
        query: 查询字符串
        docs: 文档内容列表
        doc_ids: 文档ID列表
        metadata_list: 元数据列表
        top_k: 返回结果数量
        
    返回:
        重排序后的结果列表 [(doc_id, {'content': doc, 'metadata': metadata, 'score': score}), ...]
    """
    if not docs:
        return []
        
    encoder = get_cross_encoder()
    if encoder is None:
        logging.warning("交叉编码器不可用，跳过重排序")
        # 返回原始顺序（按索引排序）
        return [(doc_id, {'content': doc, 'metadata': meta, 'score': 1.0 - idx/len(docs)}) 
                for idx, (doc_id, doc, meta) in enumerate(zip(doc_ids, docs, metadata_list))]
    
    # 准备交叉编码器输入
    cross_inputs = [[query, doc] for doc in docs]
    
    try:
        # 计算相关性得分
        scores = encoder.predict(cross_inputs)
        
        # 组合结果
        results = [
            (doc_id, {
                'content': doc, 
                'metadata': meta,
                'score': float(score)  # 确保是Python原生类型
            }) 
            for doc_id, doc, meta, score in zip(doc_ids, docs, metadata_list, scores)
        ]
        
        # 按得分排序
        results = sorted(results, key=lambda x: x[1]['score'], reverse=True)
        
        # 返回前K个结果
        return results[:top_k]
    except Exception as e:
        logging.error(f"交叉编码器重排序失败: {str(e)}")
        # 出错时返回原始顺序
        return [(doc_id, {'content': doc, 'metadata': meta, 'score': 1.0 - idx/len(docs)}) 
                for idx, (doc_id, doc, meta) in enumerate(zip(doc_ids, docs, metadata_list))]

# 新增：LLM相关性评分函数
@lru_cache(maxsize=32)
def get_llm_relevance_score(query, doc):
    """
    使用LLM对查询和文档的相关性进行评分（带缓存）
    
    参数:
        query: 查询字符串
        doc: 文档内容
        
    返回:
        相关性得分 (0-10)
    """
    try:
        # 构建评分提示词
        prompt = f"""给定以下查询和文档片段，评估它们的相关性。
        评分标准：0分表示完全不相关，10分表示高度相关。
        只需返回一个0-10之间的整数分数，不要有任何其他解释。
        
        查询: {query}
        
        文档片段: {doc}
        
        相关性分数(0-10):"""
        
        # 调用本地LLM
        response = session.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "deepseek-r1:1.5b",  # 使用较小模型进行评分
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        
        # 提取得分
        result = response.json().get("response", "").strip()
        
        # 尝试解析为数字
        try:
            score = float(result)
            # 确保分数在0-10范围内
            score = max(0, min(10, score))
            return score
        except ValueError:
            # 如果无法解析为数字，尝试从文本中提取数字
            match = re.search(r'\b([0-9]|10)\b', result)
            if match:
                return float(match.group(1))
            else:
                # 默认返回中等相关性
                return 5.0
                
    except Exception as e:
        logging.error(f"LLM评分失败: {str(e)}")
        # 默认返回中等相关性
        return 5.0

def rerank_with_llm(query, docs, doc_ids, metadata_list, top_k=5):
    """
    使用LLM对检索结果进行重排序
    
    参数:
        query: 查询字符串
        docs: 文档内容列表
        doc_ids: 文档ID列表
        metadata_list: 元数据列表
        top_k: 返回结果数量
    
    返回:
        重排序后的结果列表
    """
    if not docs:
        return []
    
    results = []
    
    # 对每个文档进行评分
    for doc_id, doc, meta in zip(doc_ids, docs, metadata_list):
        # 获取LLM评分
        score = get_llm_relevance_score(query, doc)
        
        # 添加到结果列表
        results.append((doc_id, {
            'content': doc, 
            'metadata': meta,
            'score': score / 10.0  # 归一化到0-1
        }))
    
    # 按得分排序
    results = sorted(results, key=lambda x: x[1]['score'], reverse=True)
    
    # 返回前K个结果
    return results[:top_k]

# 新增：通用重排序函数
def rerank_results(query, docs, doc_ids, metadata_list, method=None, top_k=5):
    """
    对检索结果进行重排序
    
    参数:
        query: 查询字符串
        docs: 文档内容列表
        doc_ids: 文档ID列表
        metadata_list: 元数据列表
        method: 重排序方法 ("cross_encoder", "llm" 或 None)
        top_k: 返回结果数量
        
    返回:
        重排序后的结果
    """
    # 如果未指定方法，使用全局配置
    if method is None:
        method = RERANK_METHOD
    
    # 根据方法选择重排序函数
    if method == "llm":
        return rerank_with_llm(query, docs, doc_ids, metadata_list, top_k)
    elif method == "cross_encoder":
        return rerank_with_cross_encoder(query, docs, doc_ids, metadata_list, top_k)
    else:
        # 默认不进行重排序，按原始顺序返回
        return [(doc_id, {'content': doc, 'metadata': meta, 'score': 1.0 - idx/len(docs)}) 
                for idx, (doc_id, doc, meta) in enumerate(zip(doc_ids, docs, metadata_list))]

def stream_answer(question, enable_web_search=False, model_choice="ollama"):
    """改进的流式问答处理流程，支持联网搜索、混合检索和重排序，以及多种模型选择"""
    global faiss_index # 确保可以访问
    try:
        # 检查向量数据库是否为空
        knowledge_base_exists = faiss_index is not None and faiss_index.ntotal > 0
        if not knowledge_base_exists:
                if not enable_web_search:
                    yield "⚠️ 知识库为空，请先上传文档。", "遇到错误"
                    return
                else:
                    logging.warning("知识库为空，将仅使用网络搜索结果")
        
        # progress(0.3, desc="执行递归检索...")
        # 使用递归检索获取更全面的答案上下文
        all_contexts, all_doc_ids, all_metadata = recursive_retrieval(
            initial_query=question,
            max_iterations=3,
            enable_web_search=enable_web_search,
            model_choice=model_choice
        )
        
        # 组合上下文，包含来源信息
        context_with_sources = []
        sources_for_conflict_detection = []
        
        # 使用检索到的结果构建上下文
        for doc, doc_id, metadata in zip(all_contexts, all_doc_ids, all_metadata):
            source_type = metadata.get('source', '本地文档')
            
            source_item = {
                'text': doc,
                'type': source_type
            }
            
            if source_type == 'web':
                url = metadata.get('url', '未知URL')
                title = metadata.get('title', '未知标题')
                context_with_sources.append(f"[网络来源: {title}] (URL: {url})\n{doc}")
                source_item['url'] = url
                source_item['title'] = title
            else:
                source = metadata.get('source', '未知来源')
                context_with_sources.append(f"[本地文档: {source}]\n{doc}")
                source_item['source'] = source
            
            sources_for_conflict_detection.append(source_item)
        
        # 检测矛盾
        conflict_detected = detect_conflicts(sources_for_conflict_detection)
        
        # 获取可信源
        if conflict_detected:
            credible_sources = [s for s in sources_for_conflict_detection 
                               if s['type'] == 'web' and evaluate_source_credibility(s) > 0.7]
        
        context = "\n\n".join(context_with_sources)
        
        # 添加时间敏感检测
        time_sensitive = any(word in question for word in ["最新", "今年", "当前", "最近", "刚刚"])
        
        # 改进提示词模板，提高回答质量
        prompt_template = """作为一个专业的问答助手，你需要基于以下{context_type}回答用户问题。

提供的参考内容：
{context}

用户问题：{question}

请遵循以下回答原则：
1. 仅基于提供的参考内容回答问题，不要使用你自己的知识
2. 如果参考内容中没有足够信息，请坦诚告知你无法回答,如果存在你认为和问题等价的概念和信息，可以作为答案回答
3. 回答应该全面、准确、有条理，并使用适当的段落和结构
4. 请用中文回答
5. 在回答末尾标注信息来源{time_instruction}{conflict_instruction}

请现在开始回答："""
        
        prompt = prompt_template.format(
            context_type="本地文档和网络搜索结果" if enable_web_search and knowledge_base_exists else ("网络搜索结果" if enable_web_search else "本地文档"),
            context=context if context else ("网络搜索结果将用于回答。" if enable_web_search and not knowledge_base_exists else "知识库为空或未找到相关内容。"),
            question=question,
            time_instruction="，优先使用最新的信息" if time_sensitive and enable_web_search else "",
            conflict_instruction="，并明确指出不同来源的差异" if conflict_detected else ""
        )
        
        # progress(0.7, desc="生成回答...")
        full_answer = ""
        
        # 根据模型选择使用不同的API
        if model_choice == "siliconflow":
            # 对于SiliconFlow API，不支持流式响应，所以一次性获取
            # progress(0.8, desc="通过SiliconFlow API生成回答...")
            full_answer = call_siliconflow_api(prompt, temperature=0.7, max_tokens=1536)
            
            # 处理思维链
            if "<think>" in full_answer and "</think>" in full_answer:
                processed_answer = process_thinking_content(full_answer)
            else:
                processed_answer = full_answer
                
            yield processed_answer, "完成!"
        else:
            # 使用本地Ollama模型的流式响应
            response = session.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "deepseek-r1:1.5b",
                    "prompt": prompt,
                    "stream": True
                },
                timeout=120,
                stream=True
            )
            
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode()).get("response", "")
                    full_answer += chunk
                    
                    # 检查是否有完整的思维链标签可以处理
                    if "<think>" in full_answer and "</think>" in full_answer:
                        # 需要确保完整收集一个思维链片段后再显示
                        processed_answer = process_thinking_content(full_answer)
                    else:
                        processed_answer = full_answer
                    
                    yield processed_answer, "生成回答中..."
                    
            # 处理最终输出，确保应用思维链处理
            final_answer = process_thinking_content(full_answer)
            yield final_answer, "完成!"
        
    except Exception as e:
        yield f"系统错误: {str(e)}", "遇到错误"

def query_answer(question, enable_web_search=False, model_choice="ollama"):
    """问答处理流程，支持联网搜索、混合检索和重排序，以及多种模型选择"""
    global faiss_index
    try:
        logging.info(f"收到问题：{question}，联网状态：{enable_web_search}，模型选择：{model_choice}")
        
        # 检查向量数据库是否为空
        knowledge_base_exists = faiss_index is not None and faiss_index.ntotal > 0
        if not knowledge_base_exists:
            if not enable_web_search:
                return "⚠️ 知识库为空，请先上传文档。"
            else:
                logging.warning("知识库为空，将仅使用网络搜索结果")
        
        print("执行递归检索...")
        # 使用递归检索获取更全面的答案上下文
        all_contexts, all_doc_ids, all_metadata = recursive_retrieval(
            initial_query=question,
            max_iterations=3,
            enable_web_search=enable_web_search,
            model_choice=model_choice
        )
        
        # 组合上下文，包含来源信息
        context_with_sources = []
        sources_for_conflict_detection = []
        
        for i, (context, doc_id, metadata) in enumerate(zip(all_contexts, all_doc_ids, all_metadata)):
            source_info = f"[来源 {i+1}]: {metadata.get('source', '未知')}"
            context_with_sources.append(f"{source_info}\n{context}")
            sources_for_conflict_detection.append({
                'text': context,
                'source': metadata.get('source', '未知')
            })
        
        # 检测信息冲突
        has_conflicts = detect_conflicts(sources_for_conflict_detection)
        if has_conflicts:
            print("⚠️ 检测到信息冲突，请谨慎参考")
        
        # 构建提示词
        context_text = "\n\n".join(context_with_sources)
        prompt = f"""基于以下参考信息回答问题。如果信息不足，请说明无法回答。

参考信息：
{context_text}

问题：{question}

请提供详细、准确的回答，并在必要时引用信息来源。如果发现信息冲突，请指出并说明原因。"""

        # 根据模型选择使用不同的API
        if model_choice == "siliconflow":
            # 使用SiliconFlow API
            result = call_siliconflow_api(prompt, temperature=0.7, max_tokens=1536)
            
            # 处理思维链
            processed_result = process_thinking_content(result)
            return processed_result
        else:
            # 使用本地Ollama
            response = session.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "deepseek-r1:7b",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120,
                headers={'Connection': 'close'}
            )
            response.raise_for_status()
            
            result = response.json()
            return process_thinking_content(str(result.get("response", "未获取到有效回答")))
            
    except json.JSONDecodeError:
        return "响应解析失败，请重试"
    except KeyError:
        return "响应格式异常，请检查模型服务"
    except Exception as e:
        return f"系统错误: {str(e)}"

def process_thinking_content(text):
    """处理包含<think>标签的内容，将其转换为Markdown格式"""
    # 检查输入是否为有效文本
    if text is None:
        return ""
    
    # 确保输入是字符串
    if not isinstance(text, str):
        try:
            processed_text = str(text)
        except:
            return "无法处理的内容格式"
    else:
        processed_text = text
    
    # 处理思维链标签
    try:
        while "<think>" in processed_text and "</think>" in processed_text:
            start_idx = processed_text.find("<think>")
            end_idx = processed_text.find("</think>")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                thinking_content = processed_text[start_idx + 7:end_idx]
                before_think = processed_text[:start_idx]
                after_think = processed_text[end_idx + 8:]
                
                # 使用可折叠详情框显示思维链
                processed_text = before_think + "\n\n<details>\n<summary>思考过程（点击展开）</summary>\n\n" + thinking_content + "\n\n</details>\n\n" + after_think
        
        # 处理其他HTML标签，但保留details和summary标签
        processed_html = []
        i = 0
        while i < len(processed_text):
            if processed_text[i:i+8] == "<details" or processed_text[i:i+9] == "</details" or \
               processed_text[i:i+8] == "<summary" or processed_text[i:i+9] == "</summary":
                # 保留这些标签
                tag_end = processed_text.find(">", i)
                if tag_end != -1:
                    processed_html.append(processed_text[i:tag_end+1])
                    i = tag_end + 1
                    continue
            
            if processed_text[i] == "<":
                processed_html.append("&lt;")
            elif processed_text[i] == ">":
                processed_html.append("&gt;")
            else:
                processed_html.append(processed_text[i])
            i += 1
        
        processed_text = "".join(processed_html)
    except Exception as e:
        logging.error(f"处理思维链内容时出错: {str(e)}")
        # 出错时至少返回原始文本，但确保安全处理HTML标签
        try:
            return text.replace("<", "&lt;").replace(">", "&gt;")
        except:
            return "处理内容时出错"
    
    return processed_text

def call_siliconflow_api(prompt, temperature=0.7, max_tokens=1024):
    """
    调用SiliconFlow API获取回答
    
    Args:
        prompt: 提示词
        temperature: 温度参数
        max_tokens: 最大生成token数
        
    Returns:
        生成的回答文本和思维链内容
    """
    print(f"prompt:{prompt}")
    # 检查是否配置了SiliconFlow API密钥
    if not SILICONFLOW_API_KEY:
        logging.error("未设置 SILICONFLOW_API_KEY 环境变量。请在.env文件中设置您的 API 密钥。")
        return "错误：未配置 SiliconFlow API 密钥。", ""

    try:
        payload = {
            "model": "Qwen/Qwen2.5-VL-72B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "max_tokens": max_tokens,
            "stop": None,
            "temperature": temperature,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1,
            "response_format": {"type": "text"}
        }

        headers = {
            "Authorization": f"Bearer <你的api>", # 从环境变量获取密钥
            "Content-Type": "application/json; charset=utf-8" # 明确指定编码
        }

        # 手动将payload编码为UTF-8 JSON字符串
        json_payload = json.dumps(payload, ensure_ascii=False).encode('utf-8')

        response = requests.post(
            SILICONFLOW_API_URL,
            data=json_payload, # 通过data参数发送编码后的JSON
            headers=headers,
            timeout=60  # 延长超时时间
        )

        response.raise_for_status()
        result = response.json()
        
        # 提取回答内容和思维链
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0]["message"]
            content = message.get("content", "")
            reasoning = message.get("reasoning_content", "")
            
            # 如果有思维链，则添加特殊标记，以便前端处理
            if reasoning:
                # 添加思维链标记
                full_response = f"{content}<think>{reasoning}</think>"
                return full_response
            else:
                print(f"content:{content}")
                return content
        else:
            return "API返回结果格式异常，请检查"
            
    except requests.exceptions.RequestException as e:
        logging.error(f"调用SiliconFlow API时出错: {str(e)}")
        return f"调用API时出错: {str(e)}"
    except json.JSONDecodeError:
        logging.error("SiliconFlow API返回非JSON响应")
        return "API响应解析失败"
    except Exception as e:
        logging.error(f"调用SiliconFlow API时发生未知错误: {str(e)}")
        return f"发生未知错误: {str(e)}"

from openai import OpenAI

# def call_siliconflow_api(prompt, temperature=0.7, max_tokens=1024, model="Pro/deepseek-ai/DeepSeek-R1"):
#     """
#     调用SiliconFlow API获取回答（非流式响应）
    
#     Args:
#         prompt: 提示词
#         temperature: 温度参数
#         max_tokens: 最大生成token数
#         model: 使用的模型名称
        
#     Returns:
#         生成的回答文本（包含可能的思维链内容）
#     """
#     # 检查是否配置了SiliconFlow API密钥
#     if not SILICONFLOW_API_KEY:
#         logging.error("未设置 SILICONFLOW_API_KEY 环境变量。请在.env文件中设置您的 API 密钥。")
#         return "错误：未配置 SiliconFlow API 密钥。", ""

#     try:
#         # 创建OpenAI风格的客户端
#         client = OpenAI(
#             api_key=SILICONFLOW_API_KEY,
#             base_url=SILICONFLOW_API_URL
#         )
        
#         # 创建非流式请求
#         response = client.chat.completions.create(
#             model=model,
#             messages=[
#                 {"role": "user", "content": prompt}
#             ],
#             stream=False,  # 非流式响应
#             max_tokens=max_tokens,
#             temperature=temperature,
#             top_p=0.7,
#             # top_k=50,
#             frequency_penalty=0.5
#         )
        
#         # 处理响应
#         if response.choices and len(response.choices) > 0:
#             message = response.choices[0].message
#             content = message.content or ""
            
#             # 尝试获取思维链内容（如果API支持）
#             reasoning = ""
#             if hasattr(message, 'reasoning_content') and message.reasoning_content:
#                 reasoning = message.reasoning_content
            
#             # 返回格式：(回答内容, 思维链内容)
#             return content, reasoning
    
#     except Exception as e:
#         logging.error(f"调用SiliconFlow API时出错: {str(e)}")
#         return f"API调用错误: {str(e)}", ""
    
#     # 默认返回空结果
#     return "", ""

def hybrid_merge(semantic_results, bm25_results, alpha=0.7):
    """
    合并语义搜索和BM25搜索结果
    
    参数:
        semantic_results: 向量检索结果 (字典格式，包含ids, documents, metadatas)
        bm25_results: BM25检索结果 (字典列表，包含id, score, content)
        alpha: 语义搜索权重 (0-1)
        
    返回:
        合并后的结果列表 [(doc_id, {'score': score, 'content': content, 'metadata': metadata}), ...]
    """
    merged_dict = {}
    global faiss_metadatas_map # Ensure we can access the global map
    
    # 处理语义搜索结果
    if (semantic_results and 
        isinstance(semantic_results.get('documents'), list) and len(semantic_results['documents']) > 0 and
        isinstance(semantic_results.get('metadatas'), list) and len(semantic_results['metadatas']) > 0 and
        isinstance(semantic_results.get('ids'), list) and len(semantic_results['ids']) > 0 and
        isinstance(semantic_results['documents'][0], list) and 
        isinstance(semantic_results['metadatas'][0], list) and 
        isinstance(semantic_results['ids'][0], list) and
        len(semantic_results['documents'][0]) == len(semantic_results['metadatas'][0]) == len(semantic_results['ids'][0])):
        
        num_results = len(semantic_results['documents'][0])
        # Assuming semantic_results are already ordered by relevance (higher is better)
        # A simple rank-based score, can be replaced if actual scores/distances are available and preferred
        for i, (doc_id, doc, meta) in enumerate(zip(semantic_results['ids'][0], semantic_results['documents'][0], semantic_results['metadatas'][0])):
            score = 1.0 - (i / max(1, num_results)) # Higher rank (smaller i) gets higher score
            merged_dict[doc_id] = {
                'score': alpha * score, 
                'content': doc,
                'metadata': meta
            }
    else:
        logging.warning("Semantic results are missing, have an unexpected format, or are empty. Skipping semantic part in hybrid merge.")
    
    # 处理BM25结果
    if not bm25_results:
        return sorted(merged_dict.items(), key=lambda x: x[1]['score'], reverse=True)
        
    valid_bm25_scores = [r['score'] for r in bm25_results if isinstance(r, dict) and 'score' in r]
    max_bm25_score = max(valid_bm25_scores) if valid_bm25_scores else 1.0
    
    for result in bm25_results:
        if not (isinstance(result, dict) and 'id' in result and 'score' in result and 'content' in result):
            logging.warning(f"Skipping invalid BM25 result item: {result}")
            continue
            
        doc_id = result['id']
        # Normalize BM25 score
        normalized_score = result['score'] / max_bm25_score if max_bm25_score > 0 else 0
        
        if doc_id in merged_dict:
            merged_dict[doc_id]['score'] += (1 - alpha) * normalized_score
        else:
            metadata = faiss_metadatas_map.get(doc_id, {}) # Get metadata from our global map
            merged_dict[doc_id] = {
                'score': (1 - alpha) * normalized_score,
                'content': result['content'],
                'metadata': metadata
            }
    
    merged_results = sorted(merged_dict.items(), key=lambda x: x[1]['score'], reverse=True)
    return merged_results

# 新增：更新本地文档的BM25索引
def update_bm25_index():
    """更新BM25索引，从内存中的映射加载所有文档"""
    global faiss_contents_map, faiss_id_order_for_index
    try:
        # Use the ordered list of IDs to ensure consistency
        doc_ids = faiss_id_order_for_index
        if not doc_ids:
            logging.warning("没有可索引的文档 (FAISS ID列表为空)")
            BM25_MANAGER.clear()
            return False

        # Retrieve documents in the correct order
        documents = [faiss_contents_map.get(doc_id, "") for doc_id in doc_ids]
        
        # Filter out any potential empty documents if necessary, though map access should be safe
        valid_docs_with_ids = [(doc_id, doc) for doc_id, doc in zip(doc_ids, documents) if doc]
        if not valid_docs_with_ids:
            logging.warning("没有有效的文档内容可用于BM25索引")
            BM25_MANAGER.clear()
            return False
            
        # Separate IDs and documents again for building the index
        final_doc_ids = [item[0] for item in valid_docs_with_ids]
        final_documents = [item[1] for item in valid_docs_with_ids]
            
        BM25_MANAGER.build_index(final_documents, final_doc_ids)
        logging.info(f"BM25索引更新完成，共索引 {len(final_doc_ids)} 个文档")
        return True
    except Exception as e:
        logging.error(f"更新BM25索引失败: {str(e)}")
        return False

# 新增函数：获取系统使用的模型信息
def get_system_models_info():
    """返回系统使用的各种模型信息"""
    models_info = {
        "嵌入模型": "all-MiniLM-L6-v2",
        "分块方法": "RecursiveCharacterTextSplitter (chunk_size=800, overlap=150)",
        "检索方法": "向量检索 + BM25混合检索 (α=0.7)",
        "重排序模型": "交叉编码器 (sentence-transformers/distiluse-base-multilingual-cased-v2)",
        "生成模型": "deepseek-r1 (7B/1.5B)",
        "分词工具": "jieba (中文分词)"
    }
    return models_info

# 新增函数：获取文档分块可视化数据
def get_document_chunks():
    """获取文档分块结果用于可视化"""
    global faiss_contents_map, faiss_metadatas_map, faiss_id_order_for_index
    global chunk_data_cache # Ensure we can update the global cache
    try:
        # progress(0.1, desc="正在从内存加载数据...")
        
        if not faiss_id_order_for_index:
            chunk_data_cache = []
            return [], "知识库中没有文档，请先上传并处理文档。"
            
        # progress(0.5, desc="正在组织分块数据...")
        
        doc_groups = {}
        # Iterate using the ordered IDs to reflect the FAISS index order conceptually
        for doc_id in faiss_id_order_for_index:
            doc = faiss_contents_map.get(doc_id, "")
            meta = faiss_metadatas_map.get(doc_id, {})
            if not doc: # Skip if content is somehow empty
                continue

            source = meta.get('source', '未知来源')
            if source not in doc_groups:
                doc_groups[source] = []
            
            doc_id_meta = meta.get('doc_id', '未知ID') # Get the original document ID from meta
            chunk_info = {
                "original_id": doc_id, # Keep the chunk-specific ID
                "doc_id": doc_id_meta,
                "content": doc[:200] + "..." if len(doc) > 200 else doc,
                "full_content": doc,
                "token_count": len(list(jieba.cut(doc))),
                "char_count": len(doc)
            }
            doc_groups[source].append(chunk_info)
        
        result_dicts = []
        result_lists = []
        
        # Keep track of chunks per source for indexing
        source_chunk_counters = {source: 0 for source in doc_groups.keys()}
        total_chunks = 0
        
        for source, chunks in doc_groups.items():
            num_chunks_in_source = len(chunks)
            for chunk in chunks:
                source_chunk_counters[source] += 1
                total_chunks += 1
                result_dict = {
                    "来源": source,
                    "序号": f"{source_chunk_counters[source]}/{num_chunks_in_source}",
                    "字符数": chunk["char_count"],
                    "分词数": chunk["token_count"],
                    "内容预览": chunk["content"],
                    "完整内容": chunk["full_content"],
                    "原始分块ID": chunk["original_id"] # Add original ID for potential debugging
                }
                result_dicts.append(result_dict)
                
                result_lists.append([
                    source,
                    f"{source_chunk_counters[source]}/{num_chunks_in_source}",
                    chunk["char_count"],
                    chunk["token_count"],
                    chunk["content"]
                ])
        
        # progress(1.0, desc="数据加载完成!")
        
        chunk_data_cache = result_dicts # Update the global cache
        summary = f"总计 {total_chunks} 个文本块，来自 {len(doc_groups)} 个不同来源。"
        
        return result_lists, summary
    except Exception as e:
        chunk_data_cache = []
        return [], f"获取分块数据失败: {str(e)}"

# 添加全局缓存变量
chunk_data_cache = []


# 定义函数处理事件
def clear_chat_history():
    return None, "对话已清空"

def process_chat(question, history, enable_web_search, model_choice):
    if history is None:
        history = []
    
    # 更新模型选择信息的显示
    api_text = """
    <div class="api-info" style="margin-top:10px;padding:10px;border-radius:5px;background:var(--panel-bg);border:1px solid var(--border-color);">
        <p>📢 <strong>功能说明：</strong></p>
        <p>1. <strong>联网搜索</strong>：%s</p>
        <p>2. <strong>模型选择</strong>：当前使用 <strong>%s</strong> %s</p>
    </div>
    """ % (
        "已启用" if enable_web_search else "未启用", 
        "Cloud DeepSeek-R1 模型" if model_choice == "siliconflow" else "本地 Ollama 模型",
        "(需要在.env文件中配置SERPAPI_KEY)" if enable_web_search else ""
    )
    
    # 如果问题为空，不处理
    if not question or question.strip() == "":
        history.append(("", "问题不能为空，请输入有效问题。"))
        return history, "", api_text
    
    # 添加用户问题到历史
    history.append((question, ""))
    
    # 创建生成器
    resp_generator = stream_answer(question, enable_web_search, model_choice)
    
    # 流式更新回答
    for response, status in resp_generator:
        history[-1] = (question, response)
        yield history, "", api_text

def process_chat(question, history, enable_web_search, model_choice):
    if history is None:
        history = []
    
    api_text = """
    <div class="api-info" style="margin-top:10px;padding:10px;border-radius:5px;background:var(--panel-bg);border:1px solid var(--border-color);">
        <p>📢 <strong>功能说明：</strong></p>
        <p>1. <strong>联网搜索</strong>：%s</p>
        <p>2. <strong>模型选择</strong>：当前使用 <strong>%s</strong> %s</p>
    </div>
    """ % (
        "已启用" if enable_web_search else "未启用", 
        "Cloud DeepSeek-R1 模型" if model_choice == "siliconflow" else "本地 Ollama 模型",
        "(需要在.env文件中配置SERPAPI_KEY)" if enable_web_search else ""
    )
    
    if not question or question.strip() == "":
        history.append(("", "问题不能为空，请输入有效问题。"))
        return history, "", api_text
    
    history.append((question, ""))
    
    # 这里用列表收集流式返回的内容
    full_response = ""
    resp_generator = stream_answer(question, enable_web_search, model_choice)
    for response, status in resp_generator:
        full_response = response
    
    # 更新历史回答为完整结果
    history[-1] = (question, full_response)
    
    return history, "", api_text


def update_api_info(enable_web_search, model_choice):
    api_text = """
    <div class="api-info" style="margin-top:10px;padding:10px;border-radius:5px;background:var(--panel-bg);border:1px solid var(--border-color);">
        <p>📢 <strong>功能说明：</strong></p>
        <p>1. <strong>联网搜索</strong>：%s</p>
        <p>2. <strong>模型选择</strong>：当前使用 <strong>%s</strong> %s</p>
    </div>
    """ % (
        "已启用" if enable_web_search else "未启用", 
        "Cloud DeepSeek-R1 模型" if model_choice == "siliconflow" else "本地 Ollama 模型",
        "(需要在.env文件中配置SERPAPI_KEY)" if enable_web_search else ""
    )
    return api_text

    # 绑定UI事件
    upload_btn.click(
        process_multiple_pdfs,
        inputs=[file_input],
        outputs=[upload_status, file_list],
        show_progress=True
    )

    # 绑定提问按钮
    ask_btn.click(
        process_chat,
        inputs=[question_input, chatbot, web_search_checkbox, model_choice],
        outputs=[chatbot, question_input, api_info]
    )

    # 绑定清空按钮
    clear_btn.click(
        clear_chat_history,
        inputs=[],
        outputs=[chatbot, status_display]
    )

    # 当切换联网搜索或模型选择时更新API信息
    web_search_checkbox.change(
        update_api_info,
        inputs=[web_search_checkbox, model_choice],
        outputs=[api_info]
    )
    
    model_choice.change(
        update_api_info,
        inputs=[web_search_checkbox, model_choice],
        outputs=[api_info]
    )
    
    # 新增：分块可视化刷新按钮事件
    refresh_chunks_btn.click(
        fn=get_document_chunks,
        outputs=[chunks_data, chunks_status]
    )
    
    # 新增：分块表格点击事件
    chunks_data.select(
        fn=show_chunk_details,
        inputs=chunks_data,
        outputs=chunk_detail_text
    )




# 恢复主程序启动部分
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="本地RAG问答系统")
    parser.add_argument("--files", nargs="+", help="PDF文件路径列表")
    parser.add_argument("--web-search", action="store_true", help="启用网络搜索")
    parser.add_argument("--model", choices=["ollama", "siliconflow"], default="ollama", help="选择使用的模型")
    
    args = parser.parse_args()
    if args.files:
        process_multiple_pdfs(args.files)

    process_chat("请帮我找到国家环境保护工业废水污染控制工程技术中心负责人的邮箱", None , False,"siliconflow")
    # if not check_environment():
    #     print("环境检查失败，请确保Ollama服务已启动")
    #     exit(1)
    
    # if args.files:
    #     print("开始处理文档...")
    #     result, _ = process_documents(args.files)
    #     print(result)
    
    # print("\n开始交互式问答...")
    # interactive_qa(args.web_search, args.model)

