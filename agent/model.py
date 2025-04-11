
from zhipuai import ZhipuAI
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
import torch
import re
import os

ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY", "your_zhipu_key")

class ZhipuAPIModel():
    def __init__(self, model_name):
        self.client = ZhipuAI(api_key="your_zhipu_key")  # 替换为你的 API Key