import requests

url = "https://api.siliconflow.cn/v1/chat/completions"

payload = {
    "model": "Qwen/Qwen2.5-VL-72B-Instruct",
    "stream": False,
    "max_tokens": 512,
    "enable_thinking": True,
    "thinking_budget": 4096,
    "min_p": 0.05,
    "temperature": 0.7,
    "top_p": 0.7,
    "top_k": 50,
    "frequency_penalty": 0.5,
    "n": 1,
    "stop": []
}
headers = {
    "Authorization": "Bearer sk-hjolxqxzjpqflkuivczwqujgtrmylaqocfliqdzsyszoqcia",
    "Content-Type": "application/json"
}

response = requests.request("POST", url, json=payload, headers=headers)

print(response.text)