# 1. 先在终端测试 Key 是否真的能用（最重要！！！）
import requests, os

# 把这行改成你刚复制的 Key（去掉所有空格和换行）
API_KEY = "sk-9959472de228a3d0a991911e9f30a02d8ff79862f2f1c081a84c2bda6a3b00c1"  # ← 直接粘贴

url = "https://serpapi.com/search"
payload = {"q": "test"}
headers = {"X-API-KEY": API_KEY, "Content-Type": "application/json"}

r = requests.get(url, json=payload, headers=headers)
print(r.status_code)
print(r.json())     