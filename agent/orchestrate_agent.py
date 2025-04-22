   
"""  
OrchestrateAgent: 调度和协调各种处理代理的顶层模块  
"""  

import os  
import logging  
from typing import Dict, Any, List, Optional, Union  
import json  

from .config import Config  

class OrchestrateAgent:  
    """  
    协调不同代理的顶层代理类  
    """  
    
    def __init__(self, config=None):  
        """  
        初始化协调代理  
        
        Args:  
            config: 配置对象，如果为None则使用默认配置  
        """  
        logging.basicConfig(level=logging.INFO)  
        self.logger = logging.getLogger(__name__)  
        
        # 使用提供的配置或创建默认配置  
        self.config = config if config else Config()  
        
        # 内部保存的代理实例  
        self._agents = {}  
    
    def get_long_context_agent(self, context_compression_method="semantic_compression"):  
        """  
        获取或创建长上下文代理  
        
        Args:  
            context_compression_method: 上下文压缩方法  
            
        Returns:  
            LongContextAgent实例  
        """  
        # 如果该方法的代理已经存在，直接返回  
        agent_key = f"long_context_{context_compression_method}"  
        if agent_key in self._agents:  
            return self._agents[agent_key]  
        
        # 动态导入LongContextAgent  
        from .long_context_agent import LongContextAgent  
        
        # 创建新的代理实例  
        agent = LongContextAgent(self.config, context_compression_method)  
        self._agents[agent_key] = agent  
        
        return agent  
    
    def process_document(self,   
                         document_path: str,  
                         context_compression_method: str = "semantic",  
                         output_dir: str = "output",  
                         evaluate: bool = False,  
                         run_prediction: bool = False,  
                         questions: List[str] = None) -> Dict[str, Any]:  
        """  
        处理文档  
        
        Args:  
            document_path: 文档路径  
            context_compression_method: 上下文压缩方法  
            output_dir: 输出目录  
            evaluate: 是否评估  
            run_prediction: 是否运行预测  
            questions: 问题列表（如果run_prediction为True）  
            
        Returns:  
            处理结果字典  
        """  
        self.logger.info(f"开始处理文档: {document_path}，使用方法: {context_compression_method}")  
        
        # 获取合适的代理  
        agent = self.get_long_context_agent(context_compression_method)  
        
        # 使用代理处理文档  
        result = agent.process_long_context(  
            document_path=document_path,  
            output_dir=output_dir,  
            evaluate=evaluate,  
            run_prediction=run_prediction,  
            questions=questions  
        )  
        
        self.logger.info("文档处理完成")  
        return result  
    
    def compare_methods(self,   
                        document_path: str,  
                        methods: List[str] = ["semantic", "lift"],  
                        output_base_dir: str = "comparison_output",  
                        evaluate: bool = True,  
                        questions: List[str] = None) -> Dict[str, Any]:  
        """  
        比较不同方法的效果  
        
        Args:  
            document_path: 文档路径  
            methods: 要比较的方法列表  
            output_base_dir: 输出基目录  
            evaluate: 是否评估  
            questions: 问题列表（用于评估）  
            
        Returns:  
            比较结果字典  
        """  
        self.logger.info(f"比较不同方法处理文档: {document_path}")  
        
        results = {}  
        
        for method in methods:  
            method_output_dir = os.path.join(output_base_dir, method)  
            
            # 处理文档  
            result = self.process_document(  
                document_path=document_path,  
                context_compression_method=method,  
                output_dir=method_output_dir,  
                evaluate=evaluate,  
                run_prediction=questions is not None,  
                questions=questions  
            )  
            
            results[method] = result  
        
        # 生成比较报告  
        self._generate_comparison_report(results, output_base_dir)  
        
        return results  
    
    def _generate_comparison_report(self, results: Dict[str, Dict[str, Any]], output_dir: str):  
        """生成比较报告"""  
        if not os.path.exists(output_dir):  
            os.makedirs(output_dir)  
            
        report = {  
            "methods_compared": list(results.keys()),  
            "comparison_summary": {}  
        }  
        
        # 提取比较指标  
        for method, result in results.items():  
            if "evaluation_results" in result and result["evaluation_results"]:  
                # 提取评估指标  
                metrics = {}  
                eval_results = result["evaluation_results"]  
                
                # 复制评估结果中的关键指标  
                if "compression_ratio" in eval_results:  
                    metrics["compression_ratio"] = eval_results["compression_ratio"]  
                if "perplexity" in eval_results:  
                    metrics["perplexity"] = eval_results["perplexity"]  
                if "rouge_scores" in eval_results:  
                    metrics["rouge_scores"] = eval_results["rouge_scores"]  
                    
                report["comparison_summary"][method] = metrics  
        
        # 保存比较报告  
        with open(os.path.join(output_dir, 'comparison_report.json'), 'w') as f:  
            json.dump(report, f, indent=4)  
            
        self.logger.info(f"比较报告已保存到 {os.path.join(output_dir, 'comparison_report.json')}")  
    
    
    
    
    def get_kg_agent(self):
        pass
    
    
    def get_kg_bert_agent(self):
        pass
    

    def get_kgqa_agent(self):
        pass
    
    
    
    
    def get_rbert_agent(self):
        pass
    
    
    def get_transe_agent(self):
        pass
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    pass


