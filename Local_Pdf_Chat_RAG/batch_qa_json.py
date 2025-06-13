import os
import json
from datetime import datetime
from rag_demo_pro import process_chat, process_multiple_pdfs

def load_pdf_files(pdf_files):
    """
    加载PDF文件到知识库
    
    参数:
        pdf_files: PDF文件路径列表
    返回:
        process_result: 处理结果
        file_list: 文件列表
    """
    print("开始加载PDF文件到知识库...")
    process_result, file_list = process_multiple_pdfs(pdf_files)
    print(process_result)
    return process_result, file_list

def batch_qa_from_json(input_json_path, pdf_files, enable_web_search=False, model_choice="siliconflow"):
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
    
    # 首先加载PDF文件到知识库
    load_pdf_files(pdf_files)
    
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
        
        # 调用process_chat进行问答
        history, _, _ = process_chat(question, None, enable_web_search, model_choice)
        
        # 提取回答
        if history and len(history) > 0:
            model_answer = history[-1][1]  # 获取最后一个回答
        else:
            model_answer = "未能获取回答"
        
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

def create_input_json_template(output_path="input_template.json"):
    """
    创建输入JSON模板文件
    """
    template = [
        {
            "question": "问题1",
            "answer": "标准答案1",
            "idx": 1
        },
        {
            "question": "问题2",
            "answer": "标准答案2",
            "idx": 2
        }
    ]
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    
    print(f"模板文件已创建: {output_path}")

if __name__ == "__main__":

    
    # 示例使用
    input_json_path = "./1.json"  # 替换为实际的输入JSON文件路径
    pdf_files = [
        # 在这里添加PDF文件路径
        # 例如: "./documents/1.pdf"
        "./1.pdf"
    ]
    
    try:
        results = batch_qa_from_json(
            input_json_path=input_json_path,
            pdf_files=pdf_files,
            enable_web_search=False,
            model_choice="siliconflow"
        )
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")