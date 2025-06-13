import json
import os
from datetime import datetime

def merge_json_files(file1_path, file2_path, output_dir="merged_results"):
    """
    合并两个JSON文件为一个连续的列表，idx从1开始
    
    参数:
        file1_path: 第一个JSON文件路径
        file2_path: 第二个JSON文件路径
        output_dir: 输出目录
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取两个JSON文件
    with open(file1_path, 'r', encoding='utf-8') as f1:
        data1 = json.load(f1)
    
    with open(file2_path, 'r', encoding='utf-8') as f2:
        data2 = json.load(f2)
    
    # 合并数据并重新编号
    merged_data = []
    current_idx = 1
    
    # 处理第一个文件的数据
    for item in data1:
        new_item = item.copy()
        new_item['idx'] = current_idx
        merged_data.append(new_item)
        current_idx += 1
    
    # 处理第二个文件的数据
    for item in data2:
        new_item = item.copy()
        new_item['idx'] = current_idx
        merged_data.append(new_item)
        current_idx += 1
    
    # 生成输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"merged_qa_{timestamp}.json")
    
    # 保存合并后的文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"合并完成，共 {len(merged_data)} 条数据")
    print(f"结果已保存到: {output_file}")
    return output_file

if __name__ == "__main__":
    # 示例使用
    file1_path = "Local_Pdf_Chat_RAG/1.json"
    file2_path = "Local_Pdf_Chat_RAG/2.json"
    
    merge_json_files(file1_path, file2_path) 