# 这是我们的 主Agent, 也叫编排Agent， 负责调度所有剩余的Agent

import os
import json
import ast
import subprocess
from zhipuai import ZhipuAI
from RestrictedPython import compile_restricted

# 使用时需要设置用户代理
# from googlesearch import search
from googlesearch import search as web_search

from serpapi import GoogleSearch  # 需要API key


from logging import getLogger


logger = getLogger(__name__)


# pip install zhipuai google-search-results googlesearch-python restrictedpython


class SecurityError(Exception):
    """自定义安全错误"""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
    
    def __str__(self):
        return f"安全错误, 请检查代码是否符合Python解释器的安全规范: {self.message}"
    
    def __repr__(self):
        return f"SecurityError('{self.message}')"
    
    def __eq__(self, other):
        return isinstance(other, SecurityError) and self.message == other.message


BLACKLIST = [
        'open', 'os.', 'sys.', 'exec', 'eval',
        'subprocess', '__import__', 'lambda'
    ]

class PythonExecutor:
    
    
    @staticmethod
    def safe_eval(code: str):
        for forbidden in BLACKLIST:
            if forbidden in code:
                raise SecurityError(f"检测到禁用的关键字: {forbidden}")
        try:
            # 增强安全限制
            restricted_globals = {
                "__builtins__": {
                    'None': None,
                    'True': True,
                    'False': False,
                    'sum': sum,
                    'abs': abs,
                    'max': max,
                    'min': min
                }
            }
            byte_code = compile_restricted(code, '<string>', 'eval')
            return eval(byte_code, restricted_globals, {})
        except Exception as e:
            return f"Execution Error: {str(e)}"
        
        

class ZhipuOrchestrateAgent:
    def __init__(self, model_name = "glm-4-flash"):
        self.client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))
        self.history = []
        
        self.model_name = model_name
        self.search_tool = SearchTool()  
        
        # 工具定义（符合智谱API规范）
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "访问实时网络信息，适用于需要最新数据、天气或新闻的场景",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "搜索关键词"},
                            "num_results": {"type": "number", "description": "返回结果数量(1-5)", "default": 3}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "python_executor",
                    "description": "执行数学计算或数据处理，输入应为合法的Python表达式",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "需要执行的Python代码"}
                        },
                        "required": ["code"]
                    }
                }
            }
        ]

    def _call_tool(self, tool_call):
        """工具调用分发器（安全增强版）
        
        Args:
            tool_call: 来自API的ToolCall对象
            
        Returns:
            str: 工具执行结果（确保字符串类型）
            
        Raises:
            ValueError: 参数缺失或无效时
        """
        try:
            # 统一转换为字典格式（适配智谱API规范）
            func_args = tool_call.function.arguments
            if isinstance(func_args, str):
                func_args = json.loads(func_args)

            # 工具路由逻辑
            if tool_call.function.name == "web_search":
                return self._handle_web_search(func_args)
                
            elif tool_call.function.name == "python_executor":
                return self._handle_python_code(func_args)
                
            else:
                logger.warning(f"未知工具调用: {tool_call.function.name}")
                return "[系统] 暂不支持该工具"

        except Exception as e:
            logger.error(f"工具调用失败: {str(e)}")
            return f"[系统] 工具执行错误: {str(e)}"

    def _handle_web_search(self, args: dict):
        """处理Google搜索请求（支持SerpAPI/免费方案切换）"""
        if 'query' not in args:
            raise ValueError("搜索请求缺少query参数")
            
        # 方案选择：根据是否配置SerpAPI_KEY自动切换
        if os.getenv("SERPAPI_KEY"):
            from serpapi import GoogleSearch
            
            params = {
                "q": args["query"],
                "api_key": os.getenv("SERPAPI_KEY"),
                "engine": "google",
                "num": args.get("num_results", 3)
            }
            result = GoogleSearch(params).get_dict()
            
            return "\n".join(
                f"{i+1}. {item['title']}\n{item['link']}" 
                for i, item in enumerate(result.get("organic_results", [])))
        else:
            # 免费方案（需设置合法User-Agent）
            from googlesearch import search as web_search
            
            try:
                results = list(web_search(
                    args["query"],
                    num_results=args.get("num_results", 3),
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    advanced=True
                ))
                return "\n".join(
                    f"{i+1}. {result.title}\n{result.url}" 
                    for i, result in enumerate(results))
            except Exception as e:
                raise RuntimeError(f"搜索失败: {str(e)} (可能触发反爬机制)")

    def _handle_python_code(self, args: dict):
        """处理Python代码执行（安全增强版）"""
        if 'code' not in args or not args['code'].strip():
            raise ValueError("代码执行请求缺少有效代码")
            
        # 添加自动打印最后表达式结果的逻辑
        code = args["code"].strip()
        if not code.startswith("print(") and "=" not in code:
            code = f"print({code})"
            
        # 执行安全检测
        try:
            result = PythonExecutor.safe_eval(code)
            return f"执行成功:\n{result}"
        except SyntaxError as e:
            return f"语法错误: {e.msg} (行{e.lineno})"
        
        
        
    @staticmethod
    def _format_search_result(search_data: dict) -> str:
        """将SerpAPI结果格式化为自然语言"""
        if not search_data.get("organic_results"):
            return "未找到相关结果"
        
        formatted = []
        for idx, result in enumerate(search_data["organic_results"][:5], 1):
            snippet = result.get("snippet", "暂无摘要")
            formatted.append(
                f"{idx}. 【{result['title']}】\n"
                f"链接：{result['link']}\n"
                f"摘要：{snippet}\n{'-'*30}"
            )
        return "\n".join(formatted)
    

    def chat_round(self, user_input):
        # 添加用户消息
        self.history.append({"role": "user", "content": user_input})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.history,
            tools=self.tools,
            tool_choice="auto"
        )

        message = response.choices[0].message
        self.history.append(message)

        # 处理工具调用
        if message.tool_calls:
            for tool_call in message.tool_calls:
                tool_response = self._call_tool(tool_call)
                self.history.append({
                    "role": "tool",
                    "name": tool_call.function.name,  # 新增必要字段
                    "content": tool_response,
                    "tool_call_id": tool_call.id
                })
            # 获取最终响应
            final_response = self.client.chat.completions.create(
                model="glm-4",
                messages=self.history
            )
            return final_response.choices[0].message.content
        
        return message.content



class SearchTool:
    def __init__(self, use_serpapi=False):
        self.api_key = os.getenv("SERPAPI_KEY")  # 需要单独配置
        
        self.use_serpapi = use_serpapi

    def search(self, query: str):
        
        if self.use_serpapi:
            params = {
                "q": query,
                "api_key": self.api_key,
                "engine": "google"
            }
            client = GoogleSearch(params)
            return client.get_dict()
        else:
            results = self._google_search(query)
            return results
            
    def _google_search(self, query: str):
        '''
        如果你不想使用付费的serpapi，你可以使用这个函数
        '''
        return list(web_search(
            query, 
            num_results=3,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        ))
            
    
    

if __name__ == "__main__":
    agent = ZhipuOrchestrateAgent()
    while True:
        user_input = input("用户: ")
        if user_input.lower() == 'exit':
            break
        response = agent.chat_round(user_input)
        print(f"助手: {response}")