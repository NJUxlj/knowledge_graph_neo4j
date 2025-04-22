"""  
使用压缩后的内容进行预测  
"""  

import torch  
import logging  
from typing import Dict, Any, List, Optional  
from transformers import AutoTokenizer, AutoModelForCausalLM  
import json  
import os  

class LLMPredictor:  
    """使用大型语言模型处理压缩后内容的类"""  
    
    def __init__(self, config):  
        self.config = config  
        logging.basicConfig(level=logging.INFO)  
        self.logger = logging.getLogger(__name__)  
        
        # 初始化LLM  
        self.logger.info(f"加载语言模型: {config.LLM}")  
        self.tokenizer = AutoTokenizer.from_pretrained(config.LLM)  
        self.model = AutoModelForCausalLM.from_pretrained(config.LLM)  
        self.model.to(config.DEVICE)  
    
    def generate_response(self,   
                          compressed_text: str,   
                          prompt: str,   
                          max_new_tokens: int = 512) -> str:  
        """  
        使用压缩后的内容生成回复  
        
        Args:  
            compressed_text: 压缩后的文本  
            prompt: 提示词  
            max_new_tokens: 生成的最大标记数  
            
        Returns:  
            生成的回复  
        """  
        full_prompt = f"{compressed_text}\n\n{prompt}"  
        
        self.logger.info(f"生成响应, 提示词: '{prompt}'")  
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.config.DEVICE)  
        
        with torch.no_grad():  
            output_ids = self.model.generate(  
                inputs.input_ids,  
                attention_mask=inputs.attention_mask,  
                max_new_tokens=max_new_tokens,  
                do_sample=True,  
                temperature=0.7,  
                top_p=0.9,  
                pad_token_id=self.tokenizer.eos_token_id  
            )  
        
        # 只解码新生成的标记  
        response = self.tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)  
        
        return response  
    
    def run_qa_task(self, compressed_text: str, questions: List[str]) -> Dict[str, str]:  
        """  
        运行问答任务  
        
        Args:  
            compressed_text: 压缩后的文本  
            questions: 问题列表  
            
        Returns:  
            问题和回答的字典  
        """  
        results = {}  
        
        for i, question in enumerate(questions):  
            self.logger.info(f"处理问题 {i+1}/{len(questions)}: {question}")  
            answer = self.generate_response(compressed_text, question)  
            results[question] = answer  
        
        return results  
    
    def save_prediction_results(self, results: Dict[str, Any], output_dir: str, filename: str = "prediction_results.json"):  
        """  
        保存预测结果  
        
        Args:  
            results: 预测结果  
            output_dir: 输出目录  
            filename: 输出文件名  
        """  
        if not os.path.exists(output_dir):  
            os.makedirs(output_dir)  
        
        output_path = os.path.join(output_dir, filename)  
        
        with open(output_path, 'w') as f:  
            json.dump(results, f, indent=4)  
        
        self.logger.info(f"预测结果已保存至 {output_path}")  
        
    def extract_key_information(self, compressed_text: str, num_points: int = 5) -> List[str]:  
        """  
        从压缩文本中提取关键信息点  
        
        Args:  
            compressed_text: 压缩后的文本  
            num_points: 要提取的关键点数量  
            
        Returns:  
            关键信息点列表  
        """  
        prompt = f"请从以下文本中提取{num_points}个最重要的关键信息点,以简洁的方式列出:"  
        response = self.generate_response(compressed_text, prompt)  
        
        # 简单处理，将回复拆分成列表  
        # 实际中可能需要更复杂的处理逻辑  
        key_points = [point.strip() for point in response.split('\n') if point.strip()]  
        
        return key_points  