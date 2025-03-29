from zhipuai import ZhipuAI


import os

ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY")


class ZhipuModel:
    def __init__(self, api_key = ZHIPU_API_KEY):
        self.api_key = api_key
        self.client = ZhipuAI(api_key)
        
        self.sys_prompt = "你是一个知识问答机器人，你可以回答与知识相关的问题, 并且可以熟练地整合从知识图谱中提取出的信息。"
        
        self.message = [
            {"role": "system", "content": self.sys_prompt}
        ]
        
    def generate(self, prompt, model="glm-4-fast", temperature=0.95, top_p=0.7, max_tokens=1024):

        self.message.append(
            {"role": "user", "content": prompt}
        )
        response = self.client.chat.completions.create(
            model=model,
            message = self.message,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        
        
        return response.choices[0].message.content
    
    
    