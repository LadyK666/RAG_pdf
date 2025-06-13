import openai
import re
import json
import time
from typing import List, Dict, Union
from tqdm import tqdm

class LLMJudge:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", max_retries: int = 3):
        """
        初始化评判器
        :param api_key: OpenAI API密钥
        :param model: 使用的模型名称
        :param max_retries: API调用最大重试次数
        """
        openai.api_key = api_key
        openai.base_url = "https://api.openai-hub.com/v1/"
        self.model = model
        self.max_retries = max_retries

    def _call_llm_api(self, prompt: str) -> str:
        """
        内部方法：调用LLM API
        """
        for attempt in range(self.max_retries):
            try:
                completion = openai.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a professional answer evaluator."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=500
                )
                return completion.choices[0].message.content
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"API调用失败，错误：{str(e)}")
                time.sleep(2 ** attempt)  # 指数退避

    @staticmethod
    def get_judge_instruction(question, gt_answer, pred_answer):
        return f"""
        你是一个专业的答案评估者。请评估预测答案是否正确回答了问题。
        评估标准：
        1. 预测答案不需要与标准答案完全相同，但应该在语义上等价
        2. 预测答案应该准确回答问题的核心内容
        3. 如果预测答案包含标准答案中的关键信息，即使表述不同，也应判定为正确
        
        问题：{question}
        标准答案：{gt_answer}
        预测答案：{pred_answer}
        
        请给出你的评估理由和结论。输出格式如下：
        ```json
        {{
        "rationale": "你的评估理由",
        "judgement": "correct" 或 "incorrect"
        }}
        ```
        """

    def judge_single(self, question: str, gt_answer: str, pred_answer: str) -> Dict:
        """
        单条答案评判
        """
        prompt = self.get_judge_instruction(question, gt_answer, pred_answer)
        raw_output = self._call_llm_api(prompt)
        return self.parse_output(raw_output)

    @staticmethod
    def parse_output(raw_output: str) -> Dict:
        """
        解析LLM输出
        """
        try:
            # 尝试直接解析JSON
            result = json.loads(raw_output)
        except json.JSONDecodeError:
            # 如果解析失败，使用正则表达式提取
            rationale_match = re.search(r'"rationale":\s*"([^"]*)"', raw_output)
            judgement_match = re.search(r'"judgement":\s*"(correct|incorrect)"', raw_output)
            
            result = {
                "rationale": rationale_match.group(1) if rationale_match else "无法解析理由",
                "judgement": judgement_match.group(1) if judgement_match else "invalid"
            }
        print(result)
        return result

def evaluate_qa_results(input_json_path: str, output_json_path: str, api_key: str):

    """
    评估问答结果
    :param input_json_path: 输入JSON文件路径（包含问题和模型回答）
    :param output_json_path: 输出JSON文件路径
    :param api_key: OpenAI API密钥
    """
    # 初始化评判器
    judge = LLMJudge(api_key=api_key)
    
    # 读取输入JSON文件
    with open(input_json_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    # 统计结果
    total = len(qa_data)
    correct = 0
    results = []
    
    # 使用tqdm显示进度
    for item in tqdm(qa_data, desc="评估进度"):
        question = item.get('question', '')
        standard_answer = item.get('answer', '')
        model_answer = item.get('model_answer', '')
        
        if not all([question, standard_answer, model_answer]):
            print(f"警告：跳过不完整的数据项 {item.get('idx', 'unknown')}")
            continue
        
        # 进行评判
        result = judge.judge_single(
            question=question,
            gt_answer=standard_answer,
            pred_answer=model_answer
        )
        
        # 更新统计
        if result['judgement'] == 'correct':
            correct += 1
        
        # 保存评判结果
        item['evaluation'] = result
        results.append(item)
    
    # 计算正确率
    accuracy = correct / total if total > 0 else 0
    
    # 保存结果
    output_data = {
        "summary": {
            "total_questions": total,
            "correct_answers": correct,
            "accuracy": accuracy
        },
        "detailed_results": results
    }
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n评估完成！")
    print(f"总问题数：{total}")
    print(f"正确回答数：{correct}")
    print(f"正确率：{accuracy:.2%}")
    print(f"详细结果已保存到：{output_json_path}")

if __name__ == "__main__":
    # 配置参数
    API_KEY = ""  # 替换为您的API密钥
    INPUT_JSON = "./qa_results/qa_results_20250612_100239.json"  # 替换为您的输入文件路径
    OUTPUT_JSON = "evaluation_results_2.json"  # 输出文件路径
    
    # 运行评估
    evaluate_qa_results(
        input_json_path=INPUT_JSON,
        output_json_path=OUTPUT_JSON,
        api_key=API_KEY
    ) 