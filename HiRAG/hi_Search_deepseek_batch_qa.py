import os
import logging
import numpy as np
import yaml
from hirag import HiRAG, QueryParam
from openai import AsyncOpenAI, OpenAI
from dataclasses import dataclass
from hirag.base import BaseKVStorage
from hirag._utils import compute_args_hash
from tqdm import tqdm

# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract configurations
GLM_API_KEY = config['glm']['api_key']
MODEL = config['deepseek']['model']
DEEPSEEK_API_KEY = config['deepseek']['api_key']
DEEPSEEK_URL = config['deepseek']['base_url']
GLM_URL = config['glm']['base_url']


@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)

def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro

@wrap_embedding_func_with_attrs(embedding_dim=config['model_params']['glm_embedding_dim'], max_token_size=config['model_params']['max_token_size'])
async def GLM_embedding(texts: list[str]) -> np.ndarray:
    model_name = "embedding-3"
    client = OpenAI(
        api_key=GLM_API_KEY,
        base_url=GLM_URL
    ) 
    embedding = client.embeddings.create(
        input=texts,
        model=model_name,
    )
    final_embedding = [d.embedding for d in embedding.data]
    return np.array(final_embedding)


async def deepseepk_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_URL
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------

    response = await openai_async_client.chat.completions.create(
        model=MODEL, messages=messages, **kwargs
    )

    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": MODEL}}
        )
    # -----------------------------------------------------
    return response.choices[0].message.content


graph_func = HiRAG(
    working_dir=config['hirag']['working_dir'],
    enable_llm_cache=config['hirag']['enable_llm_cache'],
    embedding_func=GLM_embedding,
    best_model_func=deepseepk_model_if_cache,
    cheap_model_func=deepseepk_model_if_cache,
    enable_hierachical_mode=config['hirag']['enable_hierachical_mode'], 
    embedding_batch_num=config['hirag']['embedding_batch_num'],
    embedding_func_max_async=config['hirag']['embedding_func_max_async'],
    enable_naive_rag=config['hirag']['enable_naive_rag'])

# comment this if the working directory has already been indexed
with open("1.txt") as f:
    graph_func.insert(f.read())

import os
import json
from datetime import datetime

def batch_qa_from_json(input_json_path):
    """
    从JSON文件读取问题并进行批量问答，将模型回答添加到JSON中
    
    参数:
        input_json_path: 输入JSON文件路径
        pdf_files: PDF文件路径列表
        enable_web_search: 是否启用网络搜索
        model_choice: 使用的模型选择 ("ollama" 或 "siliconflow")
    """
    # 创建结果保存目录
    results_dir = "qa_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    
    # 读取输入JSON文件
    with open(input_json_path, 'r', encoding='utf-8') as f:
        qa_list = json.load(f)
    
    if not isinstance(qa_list, list):
        raise ValueError("输入JSON文件必须是一个列表")
    
    # 对每个问题进行问答
    for qa_item in qa_list:
        question = qa_item.get('question')
        standard_answer = qa_item.get('answer')
        idx = qa_item.get('idx')
        
        if not question:
            print(f"警告：索引 {idx} 的问题为空，跳过")
            continue
            
        print(f"\n处理问题 {idx}: {question}")
        

        model_answer = graph_func.query(question, param=QueryParam(mode="hi"))
        # 提取回答
        # if history and len(history) > 0:
        #     model_answer = history[-1][1]  # 获取最后一个回答
        # else:
        #     model_answer = "未能获取回答"
        
        # 将模型回答添加到JSON中
        qa_item['model_answer'] = model_answer
        
        print(f"问题: {question}")
        print(f"标准答案: {standard_answer}")
        print(f"模型回答: {model_answer}")
        print("-" * 80)
    
    # 保存更新后的JSON文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(results_dir, f"qa_results_{timestamp}.json")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(qa_list, f, ensure_ascii=False, indent=2)
    
    print(f"\n所有问答已完成，结果已保存到: {output_file}")
    return qa_list

batch_qa_from_json("./1.json")
# print("Perform hi search:")
# print(graph_func.query("在污酸废水气液强化硫化处理与资源化技术中，紫金铜业项目的人力成本是每立方米多少元？", param=QueryParam(mode="hi")))
