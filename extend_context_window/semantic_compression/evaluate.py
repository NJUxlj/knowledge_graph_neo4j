"""  
评估语义压缩的效果  
"""  

import torch  
import numpy as np  
from typing import List, Dict, Any, Tuple, Optional  
import logging  
from transformers import AutoTokenizer, AutoModelForCausalLM  
from rouge_score import rouge_scorer  
from bert_score import score as bert_score  
import json  
import os  

class CompressionEvaluator:  
    """评估语义压缩效果的类"""  
    
    def __init__(self, config):  
        self.config = config  
        logging.basicConfig(level=logging.INFO)  
        self.logger = logging.getLogger(__name__)  
        
        # 初始化LLM用于计算困惑度  
        if 'perplexity' in config.EVAL_METRICS:  
            self.logger.info(f"加载语言模型: {config.LLM}")  
            self.llm_tokenizer = AutoTokenizer.from_pretrained(config.LLM)  
            self.llm_model = AutoModelForCausalLM.from_pretrained(config.LLM)  
            self.llm_model.to(config.DEVICE)  
        
        # 初始化ROUGE评估器  
        if 'rouge' in config.EVAL_METRICS:  
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)  
    
    def calculate_perplexity(self, text: str) -> float:  
        """  
        计算文本的困惑度  
        
        Args:  
            text: 待评估文本  
            
        Returns:  
            困惑度分数  
        """  
        self.logger.info("计算困惑度...")  
        # 截断文本以适应模型  
        encodings = self.llm_tokenizer(text, return_tensors="pt")  
        max_length = self.llm_model.config.max_position_embeddings  
        input_ids = encodings.input_ids[:, :max_length].to(self.config.DEVICE)  
        
        with torch.no_grad():  
            outputs = self.llm_model(input_ids, labels=input_ids)  
            
        loss = outputs.loss  
        perplexity = torch.exp(loss).item()  
        
        return perplexity  
    
    def calculate_rouge(self, original_text: str, compressed_text: str) -> Dict[str, float]:  
        """  
        计算ROUGE分数  
        
        Args:  
            original_text: 原始文本  
            compressed_text: 压缩文本  
            
        Returns:  
            ROUGE分数字典  
        """  
        self.logger.info("计算ROUGE分数...")  
        scores = self.rouge_scorer.score(original_text, compressed_text)  
        
        result = {  
            "rouge1": scores["rouge1"].fmeasure,  
            "rouge2": scores["rouge2"].fmeasure,  
            "rougeL": scores["rougeL"].fmeasure  
        }  
        
        return result  
    
    def calculate_bert_score(self, original_text: str, compressed_text: str) -> Dict[str, float]:  
        """  
        计算BERTScore  
        
        Args:  
            original_text: 原始文本  
            compressed_text: 压缩文本  
            
        Returns:  
            BERTScore字典  
        """  
        self.logger.info("计算BERTScore...")  
        P, R, F1 = bert_score([compressed_text], [original_text], lang="en")  
        
        result = {  
            "precision": P.item(),  
            "recall": R.item(),  
            "f1": F1.item()  
        }  
        
        return result  
    
    def evaluate_compression(self, original_text: str, compressed_result: Dict[str, Any]) -> Dict[str, Any]:  
        """  
        评估压缩效果  
        
        Args:  
            original_text: 原始文本  
            compressed_result: 压缩结果字典  
            
        Returns:  
            评估结果字典  
        """  
        compressed_text = compressed_result["compressed_content"]  
        evaluation_results = {  
            "compression_ratio": compressed_result["compression_ratio"],  
            "original_length": len(original_text),  
            "compressed_length": len(compressed_text),  
            "topic_count": len(compressed_result["compressed_topics"])  
        }  
        
        # 计算各项指标  
        if 'perplexity' in self.config.EVAL_METRICS:  
            try:  
                evaluation_results["perplexity"] = self.calculate_perplexity(compressed_text)  
            except Exception as e:  
                self.logger.error(f"计算困惑度时出错: {e}")  
                evaluation_results["perplexity"] = None  
        
        if 'rouge' in self.config.EVAL_METRICS:  
            try:  
                evaluation_results["rouge_scores"] = self.calculate_rouge(original_text, compressed_text)  
            except Exception as e:  
                self.logger.error(f"计算ROUGE分数时出错: {e}")  
                evaluation_results["rouge_scores"] = None  
        
        if 'bertscore' in self.config.EVAL_METRICS:  
            try:  
                evaluation_results["bert_score"] = self.calculate_bert_score(original_text, compressed_text)  
            except Exception as e:  
                self.logger.error(f"计算BERTScore时出错: {e}")  
                evaluation_results["bert_score"] = None  
        
        return evaluation_results  
    
    def save_evaluation_results(self, evaluation_results: Dict[str, Any], output_dir: str, filename: str = "evaluation_results.json"):  
        """  
        保存评估结果  
        
        Args:  
            evaluation_results: 评估结果字典  
            output_dir: 输出目录  
            filename: 输出文件名  
        """  
        if not os.path.exists(output_dir):  
            os.makedirs(output_dir)  
        
        output_path = os.path.join(output_dir, filename)  
        
        with open(output_path, 'w') as f:  
            json.dump(evaluation_results, f, indent=4)  
        
        self.logger.info(f"评估结果已保存至 {output_path}")  
        
        
        



if __name__ == '__main__':
    pass